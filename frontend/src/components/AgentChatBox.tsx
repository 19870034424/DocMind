import React, { useState, useRef, useEffect, useCallback } from 'react';
import AgentStepCard, { AgentStepData } from './AgentStepCard';
import MarkdownRenderer from './MarkdownRenderer';
import './AgentChatBox.css';

interface Source {
  heading: string;
  page: number;
  excerpt: string;
  score: number;
}

interface AgentChatBoxProps {
  docId: string;
}

type ChatMode = 'idle' | 'running' | 'clarify' | 'done';

const AgentChatBox: React.FC<AgentChatBoxProps> = ({ docId }) => {
  const [mode, setMode] = useState<ChatMode>('idle');
  const [question, setQuestion] = useState('');
  const [steps, setSteps] = useState<AgentStepData[]>([]);
  const [answerText, setAnswerText] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [followUps, setFollowUps] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [clarifyQuestion, setClarifyQuestion] = useState('');
  const [clarifyOptions, setClarifyOptions] = useState<string[]>([]);

  const abortRef = useRef<AbortController | null>(null);

  const resetState = useCallback(() => {
    setSteps([]);
    setAnswerText('');
    setSources([]);
    setFollowUps([]);
    setError(null);
    setClarifyQuestion('');
    setClarifyOptions([]);
  }, []);

  const processSSE = useCallback(async (url: string, body: any) => {
    resetState();
    setMode('running');
    setError(null);
    abortRef.current = new AbortController();

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const errMsg = (errData as any).detail || `HTTP ${response.status}`;
        setError(errMsg);
        setMode('done');
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        setError('No response body');
        setMode('done');
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              switch (event.event) {
                case 'step':
                  setSteps(prev => [...prev, event.data]);
                  if (event.data?.type === 'clarify') {
                    setClarifyQuestion(event.data.content);
                    setClarifyOptions(event.data.detail?.options || []);
                    setMode('clarify');
                  }
                  break;
                case 'chunk':
                  setAnswerText(prev => prev + (event.data?.text || ''));
                  break;
                case 'sources':
                  setSources(event.data || []);
                  break;
                case 'session':
                  if (event.data?.session_id) {
                    setSessionId(event.data.session_id);
                  }
                  break;
                case 'done':
                  setFollowUps(event.data?.follow_up_suggestions || []);
                  setMode('done');
                  break;
              }
            } catch {
              // skip malformed JSON lines
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message || 'Connection failed');
        setMode('done');
      }
    }
  }, [resetState]);

  const handleSubmit = useCallback(() => {
    if (!question.trim() || mode === 'running') return;
    processSSE(`/api/documents/${docId}/chat/deep`, { question: question.trim() });
    setQuestion('');
  }, [question, docId, mode, processSSE]);

  const handleClarifyReply = useCallback((reply: string) => {
    if (!sessionId) return;
    setMode('running');
    setClarifyQuestion('');
    setClarifyOptions([]);
    processSSE(`/api/documents/${docId}/chat/deep/resume`, {
      session_id: sessionId,
      user_response: reply,
    });
  }, [sessionId, docId, processSSE]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (mode === 'clarify') {
        handleClarifyReply(question || '请继续');
        setQuestion('');
      } else {
        handleSubmit();
      }
    }
  }, [mode, question, handleSubmit, handleClarifyReply]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  return (
    <div className="agent-chat-box">
      {/* Steps area */}
      {steps.length > 0 && (
        <div className="agent-steps">
          {steps.map((step, i) => (
            <AgentStepCard key={i} step={step} />
          ))}
        </div>
      )}

      {/* Answer area */}
      {answerText && (
        <div className="agent-answer">
          <div className="agent-answer-header">✅ 最终答案</div>
          <MarkdownRenderer content={answerText} className="agent-answer-content" />

          {sources.length > 0 && (
            <div className="agent-sources">
              <div className="agent-sources-header">📎 引用来源</div>
              {sources.map((s, i) => (
                <div key={i} className="agent-source-item">
                  <span className="source-heading">
                    {s.heading || '未知章节'}
                    {s.page > 0 ? ` (第${s.page}页)` : ''}
                  </span>
                  <span className="source-excerpt">{s.excerpt.slice(0, 150)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="agent-error">
          <span>❌ {error}</span>
          <button onClick={() => { resetState(); setMode('idle'); }}>重试</button>
        </div>
      )}

      {/* Clarify prompt */}
      {mode === 'clarify' && (
        <div className="agent-clarify">
          <div className="clarify-question">{clarifyQuestion}</div>
          {clarifyOptions.length > 0 && (
            <div className="clarify-options">
              {clarifyOptions.map((opt, i) => (
                <button key={i} onClick={() => handleClarifyReply(opt)}>
                  {opt}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Follow-up suggestions */}
      {mode === 'done' && followUps.length > 0 && (
        <div className="agent-followups">
          <span>💡 追问建议：</span>
          {followUps.map((f, i) => (
            <button key={i} onClick={() => { setQuestion(f); handleSubmit(); }}>
              {f}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="agent-input">
        <textarea
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            mode === 'clarify'
              ? '回复追问... (Enter 发送)'
              : mode === 'running'
              ? 'Agent 正在思考中...'
              : '输入问题... (Enter 发送, Shift+Enter 换行)'
          }
          disabled={mode === 'running'}
          rows={2}
        />
        <button
          onClick={mode === 'clarify' ? () => handleClarifyReply(question || '请继续') : handleSubmit}
          disabled={mode === 'running' || (!question.trim() && mode !== 'clarify')}
        >
          发送
        </button>
      </div>
    </div>
  );
};

export default AgentChatBox;
