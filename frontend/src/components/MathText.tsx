import React from 'react';
import katex from 'katex';

interface MathTextProps {
  text: string;
}

/**
 * 渲染含行内公式($...$)的纯文本, 用于表格单元格等
 * 不便走完整 markdown 渲染的场景。非公式部分原样输出。
 */
const compactLatex = (s: string) => {
  const protectedText = s.replace(/\\text\s*\{([^{}]*)\}/g, (_m, g: string) =>
    `\\text{${g.replace(/ /g, '@@SP@@')}}`
  );
  return protectedText.replace(/\s+/g, '').replace(/@@SP@@/g, ' ');
};

const MathText: React.FC<MathTextProps> = ({ text }) => {
  const parts = text.split(/(\$[^$]+\$)/g);
  return (
    <>
      {parts.map((p, i) => {
        if (p.length > 2 && p.startsWith('$') && p.endsWith('$')) {
          try {
            const html = katex.renderToString(compactLatex(p.slice(1, -1)), {
              throwOnError: false,
              output: 'html',
            });
            return <span key={i} dangerouslySetInnerHTML={{ __html: html }} />;
          } catch {
            return <span key={i}>{p}</span>;
          }
        }
        return <React.Fragment key={i}>{p}</React.Fragment>;
      })}
    </>
  );
};

export default MathText;
