import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

/**
 * MinerU 输出的块级公式常是 "$$" 独占一行且内部夹空行,
 * remark-math 遇到块内空行会解析失败, 这里剔除空行拼成连续块。
 * 同时压缩公式内部多余空格(MinerU 会输出 "\mathrm { m o d e l }"),
 * TeX 数学模式忽略空格, 压缩后 KaTeX 才能正确渲染; \text{...} 内空格保留。
 */
function normalizeMathBlocks(md: string): string {
  md = md.replace(/\$\$\s*\$\$/g, ''); // 丢弃空公式块, 避免打乱开关状态
  const protect = (s: string) => s.replace(/\\text\s*\{([^{}]*)\}/g, (_m, g) => `\\text{${g.replace(/ /g, '@@SP@@')}}`);
  const restore = (s: string) => s.replace(/@@SP@@/g, ' ');
  const compact = (s: string) => protect(s.replace(/\s+/g, ''));

  const lines = md.split('\n');
  const out: string[] = [];
  let inMath = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === '$$') {
      inMath = !inMath;
      out.push('$$');
      continue;
    }
    if (inMath) {
      if (trimmed) out.push(compact(trimmed));
    } else {
      out.push(line.replace(/\$([^$]+)\$/g, (_m, g: string) => `$${compact(g)}$`));
    }
  }
  return restore(out.join('\n'));
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className }) => {
  return (
    <div className={className ?? 'markdown-body'} style={{ lineHeight: 1.8 }}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {normalizeMathBlocks(content)}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
