# train_eval_all_methods.py Code Review Notes

在审查脚本时，发现 `argparse` 对 `--stage` 参数的取值限制有明显问题。虽然文件头部注释及阶段调度逻辑都支持 `all`，但 CLI 定义中只允许 `pre`、`lora`、`replay`、`raie`，导致使用文档推荐的 `--stage all` 时直接在解析参数阶段报错：

```
usage: train_eval_all_methods.py [-h] ...
train_eval_all_methods.py: error: argument --stage: invalid choice: 'all' (choose from 'pre', 'lora', 'replay', 'raie')
```

## 修改建议

将 `ArgumentParser.add_argument('--stage', ...)` 的 `choices` 扩展为包含 `'all'`，即可与注释及实际逻辑保持一致，例如：

```python
ap.add_argument(
    '--stage',
    type=str,
    choices=['all', 'pre', 'lora', 'replay', 'raie'],
    default='raie',
    help='选择运行阶段'
)
```

修改后，脚本允许 `--stage all`，并能正确执行 `if args.stage in ("all", ...)` 的分支逻辑，避免启动时崩溃。
