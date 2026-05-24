# Tencent Ads 2026 0.83220代码
本仓库是在腾讯广告算法大赛 2026 baseline 上逐步改出来的一版 PCVR 预测代码。相比原始 baseline，我主要围绕 loss、时间特征、候选广告建模、用户 dense 特征处理和 EMA 做了迭代。下面记录的是逐步累加的 ablation 结果。

## Score 提升路径

| 步骤 | 改动 | Score | 相对上一步 |
| --- | --- | ---: | ---: |
| Baseline | 原始 baseline | 0.8120 | - |
| 1 | 加入 Focal Loss | 0.8146 | +0.0026 |
| 2 | 加入 `day_of_week`、`hour`、`hour_sin/cos` 时间特征 | 0.8298 | +0.0152 |
| 3 | 加入 `candidate_anchor` 和 `UserDenseGroup` | 0.8303 | +0.0005 |
| 4 | 加入 EMA | 0.8322 | +0.0019 |

最终从 **0.8120 提升到 0.8322**，累计提升 **+0.0202**。

## 1. Focal Loss

**Score：0.8120 -> 0.8146，提升 +0.0026**

Baseline 使用普通 BCE。PCVR 任务里正负样本和难易样本分布不均衡，普通 BCE 容易被大量 easy sample 主导，所以我加入了 Focal Loss，让训练更关注难分样本。

具体实现：

- 在 `utils.py` 中新增 `sigmoid_focal_loss`。
- 在 `train.py` 中加入参数：
  - `--loss_type`：支持 `bce` 和 `focal`，默认使用 `focal`。
  - `--focal_alpha`：默认 `0.1`。
  - `--focal_gamma`：默认 `2.0`。
- 在 `trainer.py` 的 `_train_step` 中根据 `loss_type` 选择 BCE 或 Focal Loss。

核心逻辑是先计算逐样本 BCE，再用 focal weight 重新加权：

```python
p = torch.sigmoid(logits)
bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
p_t = p * targets + (1 - p) * (1 - targets)
focal_weight = (1 - p_t) ** gamma
alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
loss = alpha_t * focal_weight * bce_loss
```

这个改动没有改变模型结构，只替换训练目标，因此是最轻量的一步提升。

## 2. 时间特征

**Score：0.8146 -> 0.8298，提升 +0.0152**

这是提升最大的一个改动。原始 baseline 对样本发生的星期、小时等上下文时间信息建模较弱，而广告点击/转化通常有很强的日周期和周周期规律，所以我加入了样本级时间特征：

- `day_of_week`
- `hour`
- `hour_sin`
- `hour_cos`

具体实现分为数据侧和模型侧。

数据侧在 `dataset.py` 中实现 `_derive_sample_clock_features`，从样本 `timestamp` 派生时间特征：

- `day_of_week`：取值 1-7，0 作为 padding。
- `hour_id`：取值 1-24，0 作为 padding。
- `hour_sin/hour_cos`：用正余弦编码小时，保留 24 小时周期关系。
- 默认按 UTC+8 解析时间。

生成后写入 batch：

```python
result = {
    ...
    'sample_day_id': torch.from_numpy(sample_day_ids),
    'sample_hour_id': torch.from_numpy(sample_hour_ids),
    'sample_hour_sin': torch.from_numpy(sample_hour_sin),
    'sample_hour_cos': torch.from_numpy(sample_hour_cos),
}
```

模型侧在 `model.py` 中扩展 `ModelInput`，加入：

```python
sample_day_id
sample_hour_id
sample_hour_sin
sample_hour_cos
```

然后在 `PCVRHyFormer` 中新增时间 embedding 和投影层：

```python
self.day_embedding = nn.Embedding(8, emb_dim, padding_idx=0)
self.hour_embedding = nn.Embedding(25, emb_dim, padding_idx=0)
self.time_feat_proj = nn.Sequential(
    nn.Linear(2 * emb_dim + 2, d_model),
    nn.LayerNorm(d_model),
)
```

前向时通过 `_build_sample_time_bias` 将时间特征投影成 `(B, 1, D)` 的 bias：

```python
day_emb = self.day_embedding(inputs.sample_day_id.long())
hour_emb = self.hour_embedding(inputs.sample_hour_id.long())
time_feat = torch.cat([
    day_emb,
    hour_emb,
    inputs.sample_hour_sin.unsqueeze(-1),
    inputs.sample_hour_cos.unsqueeze(-1),
], dim=-1)
time_bias = F.silu(self.time_feat_proj(time_feat)).unsqueeze(1)
```

最后在 `_merge_non_sequence_tokens` 中把这个 `time_bias` 加到所有非序列 NS token 上：

```python
ns_tokens = torch.cat(ns_parts, dim=1)
return ns_tokens + time_bias
```

这样模型在看用户、广告和历史行为时，会同时知道当前样本处在一周中的哪一天、一天中的哪个小时，以及小时的周期位置。

## 3. Candidate Anchor + UserDenseGroup

**Score：0.8298 -> 0.8303，提升 +0.0005**

这一部分主要解决两个问题：

- 候选广告本身的信息需要更强地参与历史序列匹配，而不是只混在 item NS token 里。
- 用户 dense 特征中不同来源的 dense embedding 分布差异较大，直接一个线性层投影会混得太粗。

### Candidate Anchor

在 `model.py` 中新增了一条候选广告专用路径。它不直接复用混合后的 item NS token，而是从原始 item sparse/dense 特征重新构造一个纯 candidate 表示。

初始化部分：

```python
self.target_item_proj = nn.Sequential(
    nn.Linear(item_fid_count * emb_dim, d_model),
    nn.LayerNorm(d_model),
)
self.target_anchor_norm = nn.LayerNorm(d_model)
```

具体构造在 `_compose_candidate_anchor` 中完成：

- 遍历 item feature group。
- 复用 `item_ns_tokenizer` 中的 embedding table。
- 对多值 item 特征做非零 mask mean pooling。
- 将所有 item field embedding 拼接。
- 经过 `target_item_proj` 和 `target_anchor_norm` 得到 `candidate_anchor`。

`candidate_anchor` 后续在三个位置使用：

1. **Query generation**：在 `MultiSeqQueryGenerator` 中，根据 candidate 生成 `candidate_delta`，再通过 gate 调整每个序列域的 query：

```python
q_tokens = base_query + mix_gate * candidate_delta
```

2. **最终融合**：在 `_fuse_candidate_with_backbone` 中，把 backbone 输出和 candidate anchor concat 后再过融合层：

```python
torch.cat([backbone_output, candidate_anchor], dim=-1)
```

3. **历史匹配**：`CandidateHistoryMatcher` 用 candidate 作为 query 去 attend 每个历史序列，再把匹配结果作为 residual 加到最终表征上。

这个改动的直觉是：PCVR 不是只判断“用户像不像会转化”，还要判断“这个用户对当前这个广告像不像会转化”。candidate anchor 能让历史行为匹配更有目标感。

### UserDenseGroup

原始做法是把所有 user dense 特征拼成一个大向量后投影。本仓库在 `model.py` 中增加了 `UserDenseGroup` 逻辑，对 user dense 按 feature id 分组后分别投影。

分组逻辑在 `_build_user_dense_group_layout`：

```python
named_fid_groups = (
    ('emb61', {61}),
    ('emb87', {87}),
)
```

最终分成：

- `normal`
- `emb61`
- `emb87`

每组单独过 `Linear + LayerNorm`：

```python
self.user_dense_group_projs = nn.ModuleList([
    nn.Sequential(
        nn.Linear(group_dim, d_model),
        nn.LayerNorm(d_model),
    )
    for _, _, group_dim in dense_group_layout
])
```

前向时 `_build_user_dense_token` 会按 schema offset 切出对应 dense 片段，各自投影后相加，再经过 `SiLU` 作为 user dense NS token：

```python
projected_groups.append(proj(group_feats))
fused_dense = projected_groups[0]
for group_tok in projected_groups[1:]:
    fused_dense = fused_dense + group_tok
return F.silu(fused_dense).unsqueeze(1)
```

这样做可以让特殊 dense embedding 特征和普通 dense 特征先在各自空间内对齐，再融合到统一的 `d_model` 表示中。

## 4. EMA

**Score：0.8303 -> 0.8322，提升 +0.0019**

最后加入 EMA，用滑动平均后的模型参数提升验证和提交稳定性。

具体实现在 `trainer.py`：

- 默认 `ema_decay=0.999`。
- 默认 `ema_start_step=100`，前 100 step 不更新 EMA。
- 默认 `ema_update_every=1`，之后每 step 更新一次。

为了避免复制巨大的 sparse embedding 表，EMA 只维护 dense 参数：

```python
sparse_ptrs = {p.data_ptr() for p in self.model.get_sparse_params()}
for name, param in self.model.named_parameters():
    if not param.requires_grad or param.data_ptr() in sparse_ptrs:
        continue
    self.ema_shadow[name] = param.detach().clone()
```

每次 optimizer step 后调用 `_update_ema`：

```python
shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
```

验证时 `_evaluate_for_selection` 会同时评估 raw weights 和 EMA weights：

- 先跑 raw AUC / logloss。
- 再用 `_ema_scope` 临时把 dense 参数替换成 EMA 参数。
- 如果 EMA AUC 更高，就选择 EMA 结果保存 checkpoint。
- AUC 相同则用 logloss 做 tie-break。

这一步的收益主要来自降低训练后期参数抖动，让最终模型泛化更稳。

## 当前最佳结果

当前这版代码的逐步累加结果：

```text
Baseline                              0.8120
+ Focal Loss                          0.8146
+ day/hour/hour_sin/hour_cos          0.8298
+ candidate_anchor + UserDenseGroup   0.8303
+ EMA                                 0.8322
```

最终 score：**0.8322**。

## 代码文件对应关系

```text
utils.py     # sigmoid_focal_loss
train.py     # loss/time/user dense/candidate/EMA 相关参数入口
dataset.py   # timestamp -> day/hour/hour_sin/hour_cos
model.py     # PCVRHyFormer、candidate_anchor、UserDenseGroup、时间特征融合
trainer.py   # Focal Loss 选择、EMA 更新与 raw/EMA 选择保存
infer.py     # 根据 checkpoint 配置重建模型并生成 predictions.json
```
