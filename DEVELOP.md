# Development

## TorchScript

Every `torch.nn.Module` in this project must have a TorchScript conversion test.
TorchScript only supports a subset of Python and we want to make sure that all
models are convertable to TorchScript.

In this section we will give some rules of thumb to avoid conversion errors.

### Do not use global state

TorchScript cannot use global state. So, we can also not rely on `has_*` bools
in a module:

```python
class Foo(nn.Module):
    def forward(X: Tensor) -> Tensor:
        # Problem: conditional on global state.
        if has_torch_feature:
            ...
```

## Typing limitations

TorchScript only supports a small [subset of Python types](https://pytorch.org/docs/stable/jit_language_reference.html#supported-type).
This also applies to type annotations. For instance, the following will not work, because
TorchScript only supports fully-specified tuple types:

```python
class Foo(nn.Module):
    # Problem: underspecified tuple
    def shape(self) -> Tuple:
        ...

    # Problem: underspecified tuple
    def shape(self) -> Tuple[int, ...]:
        ...
```

The following is ok, because it is a valid TorchScript type:

```python
class Foo(nn.Module):
    def shape(self) -> Tuple[int, int]:
        ...
```

## Do not use `**kwargs` arguments

TorchScript does not support `**kwargs` wildcards. So the following is
invalid:

```python
class Foo(nn.Module):
    ...

    def forward(X: Tensor, **kwargs) -> Tensor:
        hidden = self.inner1(X)
        return self.inner2(hidden, **kwargs)

```

Instead we have to spell out all arguments, eg.:

```python
class Foo(nn.Module):
    ...

    def forward(X: Tensor, attention_mask: AttentionMask) -> Tensor:
        hidden = self.inner1(X)
        return self.inner2(hidden, attention_mask=attention_mask)

```
