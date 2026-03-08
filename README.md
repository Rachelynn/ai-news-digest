# 🤖 AI新闻自动收集器

自动每8小时收集AI领域最新新闻，生成摘要文档。

## 📅 更新频率

- 北京时间: 每天 08:00、16:00、00:00
- UTC时间: 每天 00:00、08:00、16:00

## 📁 文件结构

```
.
├── summaries/          # 生成的摘要文档
│   └── 2025-03-08_10-30.md
├── logs/              # 运行日志
├── collect.sh         # 主脚本
├── collect_news.py    # Python收集器
└── .github/workflows/ # GitHub Actions配置
    └── collect.yml
```

## 📖 使用方法

### 本地运行

```bash
./collect.sh
```

### 手动触发GitHub Actions

在GitHub仓库页面，点击 **Actions** → **AI News Collector** → **Run workflow**

## 📰 数据来源

- 搜索引擎实时结果
- AI行业新闻网站
- 大模型相关动态

---

*由GitHub Actions自动维护*
