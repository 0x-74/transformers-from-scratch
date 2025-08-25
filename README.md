# Decoder-Only Transformer (End-to-End)

## 1. Install `uv`
```bash
pip install uv
# or system-wide:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Install dependencies
```bash
uv sync
```

## 3. Train the model
Run training:
```bash
uv run v2.py
```

At the end of `v2.py`, save the model with a descriptive name:
```python
torch.save(model.state_dict(), "decoder_only_transformer.pth")
```

## 4. Test the model
Run testing:
```bash
uv run test.py
```

Inside `test.py`, make sure to load the correct checkpoint:
```python
model.load_state_dict(torch.load("decoder_only_transformer.pth"))
```
