from abc import ABCMeta
from itertools import chain

import numpy as np
from typing import Type, TypeVar, Literal as L, Generic, cast, Any, Optional, Annotated, get_type_hints, get_args
import numpy.typing as npt
from typing_extensions import reveal_type

Shape = tuple[int, ...]
ShapeV = TypeVar("ShapeV", bound=Shape)
ShapeV_co = TypeVar("ShapeV_co", bound=Shape, covariant=True)
DataType = npt.DTypeLike
# DataType = Type[np.dtype[Any]]
DataTypeV = TypeVar("DataTypeV", bound=DataType)
DataTypeV_co = TypeVar("DataTypeV_co", bound=DataType, covariant=True)

PropsV = TypeVar("PropsV")
Props = Annotated[Optional[PropsV], "props"]


class TensorTypeMeta(ABCMeta):
    _concrete_subclasses: dict[
        tuple[Shape, DataType],
        "Type[TensorType[Shape, DataType]]",
    ] = dict()

    _props: tuple[str, ...]
    # meta_prop: Props[DataTypeV_co] = "meta"

    def subtype(cls, s: ShapeV, t: DataTypeV) -> "Type[TensorType[ShapeV, DataTypeV]]":
        key = (s, t)
        res: Type[TensorType[ShapeV, DataTypeV]]
        try:
            res = cast(
                Type[TensorType[ShapeV, DataTypeV]],
                cls._concrete_subclasses[key],
            )
            return res
        except KeyError:
            res = cast(
                Type[TensorType[ShapeV, DataTypeV]],
                TensorTypeMeta(f"{cls.__name__}[{key}]", (cls,), {"shape": s, "dtype": t}),
            )
            cls._concrete_subclasses[key] = res
            props = tuple(
                k
                for k, v in chain(
                    get_type_hints(type(res), include_extras=True).items(),
                    get_type_hints(res, include_extras=True).items(),
                )
                if "props" in get_args(v)
            )
            res._props = props

            return res

    def __getitem__(
            self,
            item: tuple[
                Type[ShapeV] | TypeVar,
                Type[DataTypeV] | TypeVar,
            ],
    ) -> "Type[TensorType[ShapeV, DataTypeV]]":
        shape_type, dtype_type = item
        if isinstance(shape_type, TypeVar) or isinstance(dtype_type, TypeVar):
            return self  # type: ignore
        shape = tuple(
            lit.__args__[0]
            for lit in shape_type.__args__  # type: ignore
        )
        dtype = dtype_type.__args__[0]  # type:ignore
        return self.subtype(shape, dtype)  # type: ignore

    def new(cls, shape: ShapeV, dtype: DataTypeV) -> "TensorType[ShapeV, DataTypeV]":
        return cls.subtype(shape, dtype)()

    def props(cls):
        return tuple(
            getattr(cls, p)
            for p in cls._props
        )
    # results in mypy error
    # def __call__(cls, shape: ShapeV, dtype: DataTypeTV) -> "TensorType[ShapeV, DataTypeTV]":
    #     return cls.subtype(shape, dtype)()


class TensorType(Generic[ShapeV_co, DataTypeV_co], metaclass=TensorTypeMeta):
    dtype: Props[DataTypeV_co] = None
    shape: Props[ShapeV_co] = None

    def __init__(self):
        self.values = np.zeros(self.shape, self.dtype)


cls: Type[TensorType[tuple[L[2], L[5]], Type[np.float32]]] \
    = TensorType.subtype((2, 5), np.float32)

assert cls == TensorType.subtype((2, 5), np.float32)

assert issubclass(cls, TensorType)

x: TensorType[tuple[L[2], L[5]], Type[np.float32]] \
    = cls()

assert isinstance(x, cls)

assert x.values.shape == (2, 5)
assert x.dtype == np.float32

y: TensorType[tuple[L[3], L[7]], Type[np.float64]] \
    = TensorType.new((3, 7), np.float64)

# TensorType((3, 7), np.float64)  # results in mypy error


cls2: Type[TensorType[tuple[L[2], L[5]], Type[np.float32]]] \
    = TensorType[tuple[L[2], L[5]], Type[np.float32]]

assert cls2 == TensorType.subtype((2, 5), np.float32)


# correctly raises mypy error:
cls3: Type[TensorType[tuple[L[2], L[5]], Type[np.float32]]] \
    = TensorType[tuple[L[5], L[2]], Type[np.float32]]

# correctly raises mypy error:
cls5: Type[TensorType[tuple[L[2], L[5]], Type[np.float32]]] \
    = TensorType.subtype((5, 2), np.float32)
