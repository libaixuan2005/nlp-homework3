# Seq2seq德英翻译任务说明

## 任务概述

本任务实现一个基于LSTM的Seq2seq模型，将德语翻译为英语。

## 代码获取

由于完整的Seq2seq项目代码较大，请从以下仓库获取：

- **GitHub**: https://github.com/Winnie-Qi/Sequence-to-Sequence-Learning-with-LSTM/tree/main
- **Gitee**: https://gitee.com/weijie-qi/seq2seq

## 环境准备

确保已安装以下依赖包：

```bash
conda activate nlp_lbx
pip install torch torchtext spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

如果网络不好无法在线下载spacy模型，可以从学习通下载whl文件：
- `en_core_web_sm-3.7.0-py3-none-any.whl`
- `de_core_news_sm-3.7.0-py3-none-any.whl`

然后使用以下命令安装：

```bash
pip install en_core_web_sm-3.7.0-py3-none-any.whl
pip install de_core_news_sm-3.7.0-py3-none-any.whl
```

## 任务要求

1. 下载或克隆远程仓库代码到本地
2. 按照notebook中的要求完成运行
3. 在适当位置添加说明文字（使用Markdown单元格）
4. 确保所有代码单元格都有执行结果显示

## 注意事项

- 文本单元格（Markdown）在编辑完成后也需要运行（Shift+Enter）
- 每个单元格下方应该只包含一个输出结果
- 训练过程可能需要较长时间，请耐心等待
- 如果遇到内存不足，可以减小batch_size或模型参数

## 提交要求

提交包含完整执行记录的notebook文件，文件名建议为：
`task5_seq2seq_translation.ipynb`

## 参考资料

- PyTorch官方文档: https://pytorch.org/docs/stable/index.html
- Seq2seq教程: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
