#!/usr/bin/env python3
"""
AI新闻收集器 - 使用Kimi搜索API
每8小时自动收集AI新闻并生成Markdown摘要
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 添加OpenClaw路径
sys.path.insert(0, '/usr/lib/node_modules/openclaw')

def get_china_time():
    """获取北京时间"""
    return datetime.now(timezone(timedelta(hours=8)))

def format_date(dt):
    """格式化日期"""
    return dt.strftime("%Y年%m月%d日 %H:%M")

def kimi_search_news(query, limit=10):
    """使用kimi_search搜索新闻"""
    try:
        # 直接导入并调用
        from kimi_search import kimi_search
        results = kimi_search(query, limit=limit, include_content=True)
        return results if results else []
    except Exception as e:
        print(f"   搜索失败: {e}")
        return []

def search_ai_news():
    """搜索AI相关新闻"""
    queries = [
        ("AI artificial intelligence latest news today", 8),
        ("大模型 LLM OpenAI Claude Google Gemini 发布", 8),
        ("machine learning deep learning research breakthrough", 6)
    ]
    
    all_results = []
    for query, limit in queries:
        print(f"🔍 搜索: {query}")
        results = kimi_search_news(query, limit)
        print(f"   找到 {len(results)} 条结果")
        all_results.extend(results)
    
    return all_results

def generate_summary(results, source="Kimi智能搜索"):
    """生成Markdown摘要"""
    now = get_china_time()
    
    md = f"""# 🤖 AI新闻摘要

> 生成时间: {format_date(now)}  
> 来源: {source}

---

"""
    
    # 去重
    seen_urls = set()
    seen_titles = set()
    unique_results = []
    
    for item in results:
        url = item.get('url', '')
        title = item.get('title', '').strip()
        
        if url in seen_urls or title in seen_titles:
            continue
        if not title or len(title) < 5:
            continue
            
        seen_urls.add(url)
        seen_titles.add(title)
        unique_results.append(item)
        
        if len(unique_results) >= 10:
            break
    
    if not unique_results:
        md += "*暂无新内容*\n\n"
    
    for idx, item in enumerate(unique_results, 1):
        title = item.get('title', '无标题').strip()
        summary = item.get('summary', item.get('snippet', '无摘要')).strip()
        url = item.get('url', '')
        
        title = re.sub(r'\s+', ' ', title)
        summary = re.sub(r'\s+', ' ', summary)
        
        if len(summary) > 500:
            summary = summary[:500] + "..."
        
        md += f"""## {idx}. {title}

{summary}

🔗 [阅读原文]({url})

---

"""
    
    md += """
*本摘要由AI自动收集生成 | 仅供学习参考*
"""
    
    return md

def git_commit_and_push(work_dir):
    """Git提交并推送"""
    try:
        os.chdir(work_dir)
        
        result = os.popen('git remote -v').read()
        if not result.strip():
            print("⚠️ 未配置远程仓库，跳过推送")
            return False
        
        os.system('git add .')
        
        now = get_china_time()
        commit_msg = f"🤖 AI新闻更新: {now.strftime('%Y-%m-%d %H:%M')}"
        ret = os.system(f'git commit -m "{commit_msg}" > /dev/null 2>&1')
        
        if ret == 0:
            ret = os.system('git push > /dev/null 2>&1')
            if ret == 0:
                print("✅ 已推送到GitHub")
                return True
        else:
            print("ℹ️ 没有新内容需要提交")
            return True
            
    except Exception as e:
        print(f"Git操作失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='收集AI新闻')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--no-push', action='store_true', help='不推送到GitHub')
    args = parser.parse_args()
    
    work_dir = Path(__file__).parent.absolute()
    summaries_dir = work_dir / 'summaries'
    summaries_dir.mkdir(exist_ok=True)
    
    now = get_china_time()
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = summaries_dir / f"{now.strftime('%Y-%m-%d_%H-%M')}.md"
    
    print(f"🔍 开始搜索AI新闻... {format_date(now)}")
    print("-" * 50)
    
    results = search_ai_news()
    print("-" * 50)
    print(f"📰 去重后 {len(results)} 条结果，生成摘要...")
    
    summary = generate_summary(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✅ 摘要已保存: {output_path}")
    
    if not args.no_push:
        git_commit_and_push(work_dir)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
