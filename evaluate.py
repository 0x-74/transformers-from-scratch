import runpy

scripts = [
    "bigram.py",
    "v2-base.py",
    "v2-self-attn.py",
    "v2-multi-attn.py",
    "v2-ff-block.py",
    "v2-res-con.py",
    "v2-layernorm.py",
    "v2.py"
]

for script in scripts:
    print(f"Running {script}...")
    runpy.run_path(script, run_name="__main__")