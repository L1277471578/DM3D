# """
# Point, z-order and h-order are copied form PointCept
# """
# import torch
# from .hilbert import encode as hilbert_encode_
# from addict import Dict
#
# class Point(Dict):
#     """
#     Point Structure of Pointcept
#
#     A Point (point cloud) in Pointcept is a dictionary that contains various properties of
#     a batched point cloud. The property with the following names have a specific definition
#     as follows:
#
#     - "coord": original coordinate of point cloud;
#     - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
#     Point also support the following optional attributes:
#     - "offset": if not exist, initialized as batch size is 1;
#     - "batch": if not exist, initialized as batch size is 1;
#     - "feat": feature of point cloud, default input of model;
#     - "grid_size": Grid size of point cloud (related to GridSampling);
#     (related to Serialization)
#     - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
#     - "serialized_code": a list of serialization codes;
#     - "serialized_order": a list of serialization order determined by code;
#     - "serialized_inverse": a list of inverse mapping determined by code;
#     (related to Sparsify: SpConv)
#     - "sparse_shape": Sparse shape for Sparse Conv Tensor;
#     - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # If one of "offset" or "batch" do not exist, generate by the existing one
#         if "batch" not in self.keys() and "offset" in self.keys():
#             self["batch"] = offset2batch(self.offset)
#         elif "offset" not in self.keys() and "batch" in self.keys():
#             self["offset"] = batch2offset(self.batch)
#
#     def serialization(self, order="hilbert", depth=None, shuffle_orders=False):
#         """
#         Point Cloud Serialization
#
#         relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
#         """
#         assert "batch" in self.keys()
#         if "grid_coord" not in self.keys():
#             # if you don't want to operate GridSampling in data augmentation,
#             # please add the following augmentation into your pipline:
#             # dict(type="Copy", keys_dict={"grid_size": 0.01}),
#             # (adjust `grid_size` to what your want)
#             assert {"grid_size", "coord"}.issubset(self.keys())
#             self["grid_coord"] = torch.div(
#                 self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
#             ).int()
#
#         if depth is None:
#             # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
#             depth = int(self.grid_coord.max()).bit_length()
#         self["serialized_depth"] = depth
#         # Maximum bit length for serialization code is 63 (int64)
#         assert depth * 3 + len(self.offset).bit_length() <= 63
#         # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
#         # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
#         # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
#         # We can unlock the limitation by optimizing the z-order encoding function if necessary.
#         assert depth <= 16
#
#         # The serialization codes are arranged as following structures:
#         # [Order1 ([n]),
#         #  Order2 ([n]),
#         #   ...
#         #  OrderN ([n])] (k, n)
#         code = [
#             encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
#         ]
#         code = torch.stack(code)
#         order = torch.argsort(code)
#         inverse = torch.zeros_like(order).scatter_(
#             dim=1,
#             index=order,
#             src=torch.arange(0, code.shape[1], device=order.device).repeat(
#                 code.shape[0], 1
#             ),
#         )
#
#         if shuffle_orders:
#             perm = torch.randperm(code.shape[0])
#             code = code[perm]
#             order = order[perm]
#             inverse = inverse[perm]
#
#         self["serialized_code"] = code
#         self["serialized_order"] = order
#         self["serialized_inverse"] = inverse
#
# @torch.inference_mode()
# def offset2bincount(offset):
#     return torch.diff(
#         offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
#     )
#
# @torch.inference_mode()
# def offset2batch(offset):
#     bincount = offset2bincount(offset)
#     return torch.arange(
#         len(bincount), device=offset.device, dtype=torch.long
#     ).repeat_interleave(bincount)
#
# @torch.inference_mode()
# def batch2offset(batch):
#     return torch.cumsum(batch.bincount(), dim=0).long()
#
# @torch.inference_mode()
# def encode(grid_coord, batch=None, depth=16, order="hilbert"):
#     if order == "hilbert":
#         code = hilbert_encode(grid_coord, depth=depth)
#     elif order == "hilbert-trans":
#         code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
#     else:
#         raise NotImplementedError
#     if batch is not None:
#         batch = batch.long()
#         code = batch << depth * 3 | code
#     return code
#
# def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
#     return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)



import torch
import spconv.pytorch as spconv

try:
    import ocnn
except ImportError:
    ocnn = None
from addict import Dict
from typing import List

from models.serialization import encode
# from models.utils import (
#     offset2batch,
#     batch2offset,
#     offset2bincount,
#     bincount2offset,
# )


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加batch张量检查

        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    # 在文档1的Point类中添加检查方法
    def validate_point_structure(self):
        assert "batch" in self, "Missing batch information"

        if "grid_coord" in self:
            n_points = len(self.grid_coord)
        elif "coord" in self:
            n_points = len(self.coord)
        else:
            raise ValueError("Missing spatial coordinates (coord/grid_coord)")

        assert len(self.batch) == n_points, (
            f"Batch size ({len(self.batch)}) doesn't match point count ({n_points})! "
            "Regenerate batch indices with offset2batch."
        )

        # 可选：验证offset一致性
        if "offset" in self:
            expected_points = self.offset[-1] if len(self.offset) > 0 else 0
            assert n_points == expected_points, (
                f"Offset claims {expected_points} points, actual {n_points}"
            )


    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        self["order"] = order
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())

            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max() + 1).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def octreelization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        relay on ["grid_coord", "batch", "feat"]
        """
        assert (
            ocnn is not None
        ), 
        assert {"feat", "batch"}.issubset(self.keys())
        # add 1 to make grid space support shift order
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 1
        self["depth"] = depth
        assert depth <= 16  # maximum in ocnn

        # [0, 2**depth] -> [0, 2] -> [-1, 1]
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()

        query_pts = torch.cat([self.grid_coord, point.batch_id], dim=1).contiguous()
        inverse = octree.search_xyzb(query_pts, depth, True)
        assert torch.sum(inverse < 0) == 0  # all mapping should be valid
        inverse_ = torch.unique(inverse)
        order = torch.zeros_like(inverse_).scatter_(
            dim=0,
            index=inverse,
            src=torch.arange(0, inverse.shape[0], device=inverse.device),
        )
        self["octree"] = octree
        self["octree_order"] = order
        self["octree_inverse"] = inverse

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)

@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

