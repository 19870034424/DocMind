import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";

interface Props {
  content: string;
}

const schema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames || []), "div", "span", "center"],
  attributes: {
    ...defaultSchema.attributes,
    "*": ["className", "id", "style"],
  },
};

export default function ReportViewer({ content }: Props) {
  return (
    <div className="report-viewer">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeSanitize, schema]]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
