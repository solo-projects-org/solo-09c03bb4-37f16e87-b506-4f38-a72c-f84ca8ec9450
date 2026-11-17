"""编号脱敏：把反例中的订单号/退款号/退款归属号换成与原号不重叠的正整数。

用途：结算数仓工程师要把真实的增量净额错算反例交给外部 SQL 维护者，
但不能泄露真实业务编号。本模块负责：

1. **确认原样本有差异**：用调用方提供的执行器按原 SQL 重放原始输入，
   只有出现差异（错算反例成立）才继续；无法判定（报错/超时/超限）或无
   差异都拒绝交付。
2. **构造编号映射**：收集初始数据与所有批次前后像中出现的订单号、退款号、
   退款归属号（``refunds.order_id`` 与 ``orders.order_id`` 同号同源，按
   数值一致映射），统一替换成与原号不重叠的正整数。映射是单射，且新旧
   编号集合不相交，因此相等性结构（含跨表比较）完全保留。金额、批号、
   变更顺序保持不变。
3. **保真验证**：脱敏样本用**原 SQL** 再重放一次。只有
   - 首次差异批号与原样本相同，且
   - 缺失/多余行按编号映射后与原见证逐项一致（多重集比较），
   才判定为可分享。任何一步无法判定或不一致，都只报告"无法安全分享"，
   不交付部分样本。

交付物（脱敏初态、批次、新见证）中不含原 SQL、原号映射与原始编号。
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import config
from .executor import (
    OUTCOME_DIFFERENT,
    OUTCOME_ERROR,
    OUTCOME_TIMEOUT,
    Executor,
)
from .models import Batch, ChangeOp, InitialData, OrderRow, RefundRow
from .validate import InputError, validate_all

STATUS_SHAREABLE = "shareable"
STATUS_NOT_SHAREABLE = "not_shareable"

#: 前后像中承载业务编号的字段（订单号、退款号、退款归属号）。
_ID_FIELDS = ("order_id", "refund_id")


@dataclass
class ShareAttempt:
    """一次脱敏分享尝试的结果。失败时所有样本字段均为 None。"""

    status: str
    initial: InitialData | None = None
    batches: list[Batch] | None = None
    first_difference_batch_id: int | None = None
    difference: tuple[list[list[Any]], list[list[Any]]] | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 编号收集与映射
# ---------------------------------------------------------------------------

def collect_ids(initial: InitialData, batches: Iterable[Batch]) -> set[int]:
    """收集初态与所有批次前后像中出现的全部业务编号。"""
    ids: set[int] = set()
    for row in initial.orders:
        ids.add(row.order_id)
    for row in initial.refunds:
        ids.add(row.refund_id)
        ids.add(row.order_id)
    for batch in batches:
        for change in batch.changes:
            for image in (change.before, change.after):
                if not isinstance(image, dict):
                    continue
                for key in _ID_FIELDS:
                    value = image.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        ids.add(value)
    return ids


def build_mapping(ids: set[int]) -> dict[int, int]:
    """为每个原号分配一个与原号集合不重叠的正整数。

    取"未被原号占用的最小正整数"，按原号升序依次分配：单射、确定、
    有界（不超过 2×编号个数，不会溢出 64 位），且新号集合与原号集合
    不相交。同一数值在订单号/退款号/退款归属号之间共享同一映射，
    因此跨表的相等性关系（如 ``order_id = refund_id`` 这类比较）也
    被完整保留。
    """
    mapping: dict[int, int] = {}
    candidate = 1
    for old in sorted(ids):
        while candidate in ids:
            candidate += 1
        mapping[old] = candidate
        candidate += 1
    return mapping


def _map_image(
    image: dict[str, Any] | None, mapping: dict[int, int]
) -> dict[str, Any] | None:
    """替换一个前后像中的编号字段；金额等其他字段原样保留。"""
    if image is None:
        return None
    mapped = dict(image)
    for key in _ID_FIELDS:
        value = mapped.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in mapping
        ):
            mapped[key] = mapping[value]
    return mapped


def anonymize_initial(initial: InitialData, mapping: dict[int, int]) -> InitialData:
    return InitialData(
        orders=[
            OrderRow(order_id=mapping[row.order_id], amount=row.amount)
            for row in initial.orders
        ],
        refunds=[
            RefundRow(
                refund_id=mapping[row.refund_id],
                order_id=mapping[row.order_id],
                amount=row.amount,
            )
            for row in initial.refunds
        ],
    )


def anonymize_batches(
    batches: Iterable[Batch], mapping: dict[int, int]
) -> list[Batch]:
    """替换每批每个变更前后像中的编号；批号与变更顺序保持不变。"""
    anonymized: list[Batch] = []
    for batch in batches:
        changes = [
            ChangeOp(
                op=change.op,
                table=change.table,
                before=_map_image(change.before, mapping),
                after=_map_image(change.after, mapping),
            )
            for change in batch.changes
        ]
        anonymized.append(Batch(batch_id=batch.batch_id, changes=changes))
    return anonymized


# ---------------------------------------------------------------------------
# 见证对应检查
# ---------------------------------------------------------------------------

def _map_witness_rows(
    rows: Iterable[list[Any]], mapping: dict[int, int]
) -> list[tuple[Any, Any]] | None:
    """把原见证行的订单号（第 1 列）按映射换成新号；净额（第 2 列）不变。

    编号映射只覆盖输入中出现的编号；SQL 常量等未映射值保持原样
    （两份样本用同一条 SQL 重放，常量产生的行本就一致）。
    """
    mapped: list[tuple[Any, Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return None
        order_id, net_amount = row
        try:
            mapped_order_id = mapping.get(order_id, order_id)
        except TypeError:  # pragma: no cover - 防御不可哈希值
            mapped_order_id = order_id
        mapped.append((mapped_order_id, net_amount))
    return mapped


def _witness_matches(
    original_rows: list[list[Any]],
    new_rows: list[list[Any]],
    mapping: dict[int, int],
) -> bool:
    """原见证按编号映射后是否与新见证一致（多重集比较）。"""
    mapped = _map_witness_rows(original_rows, mapping)
    if mapped is None:
        return False
    new_tuples: list[tuple[Any, ...]] = []
    for row in new_rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return False
        new_tuples.append(tuple(row))
    try:
        return Counter(mapped) == Counter(new_tuples)
    except TypeError:  # pragma: no cover - 防御不可哈希值
        return False


# ---------------------------------------------------------------------------
# 脱敏分享主流程
# ---------------------------------------------------------------------------

def attempt_share(
    initial: InitialData,
    batches: list[Batch],
    executor: Executor,
    *,
    deadline: float | None = None,
) -> ShareAttempt:
    """确认差异 → 脱敏 → 双重重放验证，全部通过才交付脱敏样本。

    任一环节无法判定或不保真，都返回 ``not_shareable`` 且不携带任何
    样本内容（不交付部分样本）。说明文字只含批号与计数，不含原始编号、
    不含 SQL 文本。
    """
    if deadline is None:
        deadline = time.monotonic() + config.MAX_REQUEST_SECONDS

    # 1) 先确认原样本确实有差异（错算反例成立）。
    original = executor.run(initial, batches, deadline=deadline)
    if original.outcome in (OUTCOME_ERROR, OUTCOME_TIMEOUT):
        return ShareAttempt(
            STATUS_NOT_SHAREABLE,
            notes=[
                "原样本重放无法判定（SQL 报错、被安全策略拒绝或超时），"
                "无法安全分享"
            ],
        )
    if original.outcome != OUTCOME_DIFFERENT or original.first_difference is None:
        return ShareAttempt(
            STATUS_NOT_SHAREABLE,
            notes=["原样本重放未发现差异，没有可分享的错算反例"],
        )
    original_diff = original.first_difference

    # 2) 构造与原号不重叠的编号映射并脱敏。
    mapping = build_mapping(collect_ids(initial, batches))
    anon_initial = anonymize_initial(initial, mapping)
    anon_batches = anonymize_batches(batches, mapping)
    try:
        validate_all(anon_initial, anon_batches)
    except InputError:  # pragma: no cover - 单射映射保持合法性，防御分支
        return ShareAttempt(
            STATUS_NOT_SHAREABLE,
            notes=["脱敏后的样本未通过合法性校验，无法安全分享"],
        )

    # 3) 脱敏样本用原 SQL 重放，核对首次差异批号与见证。
    replayed = executor.run(anon_initial, anon_batches, deadline=deadline)
    if replayed.outcome in (OUTCOME_ERROR, OUTCOME_TIMEOUT):
        return ShareAttempt(
            STATUS_NOT_SHAREABLE,
            notes=[
                "脱敏样本重放无法判定（SQL 报错、被安全策略拒绝或超时），"
                "无法安全分享"
            ],
        )
    if replayed.outcome != OUTCOME_DIFFERENT or replayed.first_difference is None:
        return ShareAttempt(
            STATUS_NOT_SHAREABLE,
            notes=["脱敏样本未重现原样本的差异，无法安全分享"],
        )
    new_diff = replayed.first_difference
    if new_diff.batch_id != original_diff.batch_id:
        return ShareAttempt(
            STATUS_NOT_SHAREABLE,
            notes=["脱敏样本的首次差异批号与原样本不一致，无法安全分享"],
        )
    if not _witness_matches(
        original_diff.missing, new_diff.missing, mapping
    ) or not _witness_matches(original_diff.extra, new_diff.extra, mapping):
        return ShareAttempt(
            STATUS_NOT_SHAREABLE,
            notes=["脱敏样本的缺失/多余行与原见证按编号不对应，无法安全分享"],
        )

    return ShareAttempt(
        STATUS_SHAREABLE,
        initial=anon_initial,
        batches=anon_batches,
        first_difference_batch_id=new_diff.batch_id,
        difference=(new_diff.missing, new_diff.extra),
        notes=[
            f"原样本在批次 {original_diff.batch_id} 首次出现差异"
            f"（缺 {len(original_diff.missing)} 行、多 {len(original_diff.extra)} 行）",
            "已将初态与全部前后像中的订单号、退款号、退款归属号一致替换为"
            "与原号不重叠的正整数；金额、批号与变更顺序保持不变",
            "脱敏样本已用原 SQL 重放验证：首次差异批号相同，"
            "缺失/多余行按编号对应后与原见证一致",
            "交付内容不含原 SQL、原号映射与原始编号",
        ],
    )
