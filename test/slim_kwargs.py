__docformat__ = "restructuredtext"
__all__ = ["SLIM_KWARGS", "ALL_TRUE_SLIM_KWARGS"]

import types
from collections import defaultdict

ALL_TRUE_SLIM_KWARGS = types.MappingProxyType(
    {
        "constant_to_initializer": True,
        "fuse_constant_nodes": True,
        "fuse_matmul_add": True,
        "fuse_gemm_reshape_bn": True,
        "fuse_bn_reshape_gemm": True,
        "fuse_bn_gemm": True,
        "fuse_transpose_bn_transpose": True,
        "fuse_gemm_gemm": True,
        "fuse_conv_bn": True,
        "fuse_bn_conv": True,
        "fuse_convtransposed_bn": True,
        "simplify_conv_to_flatten_gemm": True,
        "simplify_gemm": True,
        "remove_redundant_operations": True,
        "simplify_node_name": True,
        "reorder_by_strict_topological_order": True,
    }
)

acasxu_2023 = types.MappingProxyType(
    {
        "fuse_matmul_add": True,
        "remove_redundant_operations": True,
    }
)
cctsdb_yolo_2023 = types.MappingProxyType(
    {"fuse_constant_nodes": True, "has_batch_dim": False}
)
cifar100_2024 = types.MappingProxyType(
    {
        "fuse_conv_bn": True,
        "fuse_bn_conv": True,
    }
)
cgan_2023 = types.MappingProxyType(
    {
        "fuse_conv_bn": True,
        "fuse_bn_conv": True,
        "fuse_convtransposed_bn": True,
        "fuse_constant_nodes": True,
        "remove_redundant_operations": True,
    }
)
collins_aerospace_benchmark = types.MappingProxyType({})
collins_rul_cnn_2022 = types.MappingProxyType(
    {
        "simplify_conv_to_flatten_gemm": True,
        "remove_redundant_operations": True,
    }
)
cora_2024 = types.MappingProxyType({"fuse_matmul_add": True})
dist_shift_2023 = types.MappingProxyType({"remove_redundant_operations": True})
linearizenn = types.MappingProxyType({})
lsnc = types.MappingProxyType({"fuse_constant_nodes": True})
metaroom_2023 = types.MappingProxyType({})
ml4acopf_2024 = types.MappingProxyType({"fuse_constant_nodes": True})
nn4sys = types.MappingProxyType({"fuse_matmul_add": True, "simplify_gemm": True})
safenlp_2024 = types.MappingProxyType({"fuse_matmul_add": True})
tinyimagenet_2024 = types.MappingProxyType({"fuse_conv_bn": True})
tllverifybench_2023 = types.MappingProxyType({"fuse_matmul_add": True})
traffic_signs_recognition_2023 = types.MappingProxyType({})
vggnet16_2022 = types.MappingProxyType({})
vit_2023 = types.MappingProxyType(
    {
        # TODO: There is a bug for gemm with multiple dimensions.
        "fuse_constant_nodes": True,
        "fuse_matmul_add": True,
        "fuse_transpose_bn_transpose": True,
        "fuse_gemm_gemm": True,
        "fuse_bn_gemm": True,
        "remove_redundant_operations": True,
    }
)
yolo_2023 = types.MappingProxyType({})

test = types.MappingProxyType(
    {
        "fuse_matmul_add": True,
        "remove_redundant_operations": True,
        "has_batch_dim": False,
    }
)


cersyve = types.MappingProxyType({"fuse_gemm_gemm": True})
lsnc_relu = types.MappingProxyType({"fuse_constant_nodes": True})
malbeware = types.MappingProxyType({})
relusplitter = types.MappingProxyType({})
sat_relu = types.MappingProxyType({})
soundnessbench = types.MappingProxyType({})

SLIM_KWARGS = defaultdict(
    lambda: None,
    types.MappingProxyType(
        {
            "acasxu_2023": acasxu_2023,
            "cctsdb_yolo_2023": cctsdb_yolo_2023,
            "cersyve": cersyve,
            "cgan_2023": cgan_2023,
            "cifar100": cifar100_2024,
            "cifar100_2024": cifar100_2024,
            "collins_aerospace_benchmark": collins_aerospace_benchmark,
            "collins_rul_cnn_2022": collins_rul_cnn_2022,
            "collins_rul_cnn_2023": collins_rul_cnn_2022,
            "cora": cora_2024,
            "cora_2024": cora_2024,
            "dist_shift_2023": dist_shift_2023,
            "linearizenn": linearizenn,
            "lsnc": lsnc,
            "lsnc_relu": lsnc_relu,
            "malbeware": malbeware,
            "metaroom_2023": metaroom_2023,
            "ml4acopf": ml4acopf_2024,
            "ml4acopf_2023": ml4acopf_2024,
            "ml4acopf_2024": ml4acopf_2024,
            "nn4sys": nn4sys,
            "nn4sys_2023": nn4sys,
            "relusplitter": relusplitter,
            "safenlp": safenlp_2024,
            "safenlp_2024": safenlp_2024,
            "sat_relu": sat_relu,
            "soundnessbench": soundnessbench,
            "tinyimagenet": tinyimagenet_2024,
            "tinyimagenet_2024": tinyimagenet_2024,
            "tllverifybench_2023": tllverifybench_2023,
            "traffic_signs_recognition_2023": traffic_signs_recognition_2023,
            "vggnet16_2022": vggnet16_2022,
            "vggnet16_2023": vggnet16_2022,
            "vit_2023": vit_2023,
            "yolo_2023": yolo_2023,
            "test": ALL_TRUE_SLIM_KWARGS,
        },
    ),
)
