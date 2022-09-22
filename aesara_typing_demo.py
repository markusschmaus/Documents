import numpy as np
from typing import Type, TypeVar, Literal as L, Generic, cast
import numpy.typing as npt

Shape = tuple[int, ...]
ShapeV = TypeVar("ShapeV", bound=Shape)
ShapeV_co = TypeVar("ShapeV_co", bound=Shape, covariant=True)
DataType = npt.DTypeLike
DataTypeV = TypeVar("DataTypeV", bound=DataType)
DataTypeV_co = TypeVar("DataTypeV_co", bound=DataType, covariant=True)


class TensorTypeMeta(type):
    _concrete_subclasses: dict[
        tuple[Shape, DataType],
        "Type[TensorType[Shape, DataType]]",
    ] = dict()

    def subclass(cls, s: ShapeV, t: DataTypeV) -> "Type[TensorType[ShapeV, DataTypeV]]":
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
                ConcreteTensorTypeMeta(
                    f"{cls.__name__}[{key}]",
                    (cls,),
                    {"shape": s, "dtype": t}
                ),
            )
            cls._concrete_subclasses[key] = res
            return res

    def new(cls, shape: ShapeV, dtype: DataTypeV) -> "TensorType[ShapeV, DataTypeV]":
        return cls.subclass(shape, dtype)()

    # results in mypy error
    # def __call__(cls, shape: ShapeV, dtype: DataTypeTV) -> "TensorType[ShapeV, DataTypeTV]":
    #     return cls.subclass(shape, dtype)()


class ConcreteTensorTypeMeta(TensorTypeMeta):
    # Not necessary if __call__ isn't overridden in TensorTypeMeta
    # def __call__(cls, *args, **kwargs):
    #     obj = cls.__new__(cls, *args, **kwargs)
    #     obj.__init__(*args, **kwargs)
    #     return obj
    pass


class TensorType(Generic[ShapeV_co, DataTypeV_co], metaclass=TensorTypeMeta):
    dtype: DataTypeV_co
    shape: ShapeV_co

    def __init__(self):
        self.values = np.zeros(self.shape, self.dtype)


cls: Type[TensorType[tuple[L[2], L[5]], Type[np.float32]]] \
    = TensorType.subclass((2, 5), np.float32)

assert cls == TensorType.subclass((2, 5), np.float32)

x: TensorType[tuple[L[2], L[5]], Type[np.float32]] \
    = cls()

assert x.values.shape == (2, 5)
assert x.dtype == np.float32

y: TensorType[tuple[L[3], L[7]], Type[np.float64]] \
    = TensorType.new((3, 7), np.float64)

# TensorType((3, 7), np.float64)  # results in mypy error
