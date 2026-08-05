import React, { useState } from 'react';
import './AgentStepCard.css';

export interface AgentStepData {
  step_index: number;
  type: 'plan' | 'tool_call' | 'tool_result' | 'reflect' | 'clarify' | 'answer' | 'error';
  content: string;
  detail?: Record<string, any> | null;
}

interface AgentStepCardProps {
  step: AgentStepData;
}

const ICON_MAP: Record<string, string> = {
  plan: '📋',
  tool_call: '🔍',
  tool_result: '📄',
  reflect: '🤔',
  clarify: '💬',
  answer: '✅',
  error: '❌',
};

const LABEL_MAP: Record<string, string> = {
  plan: '规划',
  tool_call: '工具调用',
  tool_result: '检索结果',
  reflect: '反思验证',
  clarify: '追问',
  answer: '回答',
  error: '错误',
};

const AgentStepCard: React.FC<AgentStepCardProps> = ({ step }) => {
  const [expanded, setExpanded] = useState(false);
  const icon = ICON_MAP[step.type] || '•';
  const label = LABEL_MAP[step.type] || step.type;

  const hasDetail = step.detail && Object.keys(step.detail).length > 0;

  let detailContent: string | null = null;
  if (hasDetail && step.detail) {
    if (step.type === 'tool_call') {
      detailContent = `工具: ${step.detail.tool_name || ''}\n参数: ${JSON.stringify(step.detail.arguments || {}, null, 2)}`;
    } else if (step.type === 'tool_result') {
      detailContent = step.detail.result_summary || step.content;
    } else {
      detailContent = JSON.stringify(step.detail, null, 2);
    }
  }

  return (
    <div className={`agent-step-card step-${step.type}`}>
      <div className="step-header" onClick={() => setExpanded(!expanded)}>
        <span className="step-icon">{icon}</span>
        <span className="step-label">{label}</span>
        <span className="step-content-preview">
          {step.content.length > 80 ? step.content.slice(0, 80) + '...' : step.content}
        </span>
        {hasDetail && (
          <span className="step-expand-icon">{expanded ? '▾' : '▸'}</span>
        )}
      </div>
      {expanded && hasDetail && (
        <div className="step-detail">
          <pre>{detailContent}</pre>
        </div>
      )}
      {expanded && !hasDetail && (
        <div className="step-detail">
          <p>{step.content}</p>
        </div>
      )}
    </div>
  );
};

export default AgentStepCard;
