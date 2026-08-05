import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage, askQuestionStream, getChatHistory, clearChatHistory } from "../services/api";
import "./ChatBox.css";

interface Props {
  docId: string;
}

export default function ChatBox({ docId }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getChatHistory(docId).then(setMessages).catch(() => {});
  }, [docId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const question = input.trim();
    if (!question || loading) return;
    setInput("");
    setError("");
    const userMsg: ChatMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMsg, { role: "assistant", content: "" }]);
    setLoading(true);
    try {
      await askQuestionStream(
        docId,
        question,
        (text) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === "assistant") {
              updated[updated.length - 1] = { ...last, content: last.content + text };
            }
            return updated;
          });
        },
      );
    } catch (e: any) {
      setError(e.message || "发送失败");
      // Remove the empty assistant message and user message on error
      setMessages((prev) => prev.slice(0, -2));
    } finally {
      setLoading(false);
    }
  }

  async function handleClear() {
    await clearChatHistory(docId);
    setMessages([]);
  }

  return (
    <div className="chat-box">
      <div className="chat-header">
        <span className="chat-title">文档问答</span>
        {messages.length > 0 && (
          <button className="btn btn-sm btn-outline" onClick={handleClear}>
            清空对话
          </button>
        )}
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <p className="chat-empty">针对文档内容提问，AI 将从文档中检索相关段落后回答。</p>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`chat-msg ${msg.role}`}>
              <span className="chat-role">{msg.role === "user" ? "你" : "AI"}</span>
              <div className="chat-bubble">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              </div>
            </div>
          ))
        )}
        {loading && messages[messages.length - 1]?.content === "" && (
          <div className="chat-msg assistant">
            <span className="chat-role">AI</span>
            <div className="chat-bubble typing">思考中...</div>
          </div>
        )}
        {error && <p className="chat-error">{error}</p>}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="输入你的问题..."
          disabled={loading}
          maxLength={2000}
        />
        <button className="btn" onClick={handleSend} disabled={loading || !input.trim()}>
          发送
        </button>
      </div>
    </div>
  );
}
