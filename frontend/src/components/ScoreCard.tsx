import "./ScoreCard.css";

interface Props {
  content: string;
}

interface Scores {
  evidence: number;
  logic: number;
  data: number;
}

function extractScores(markdown: string): Scores | null {
  const patterns: Record<string, RegExp[]> = {
    evidence: [
      /论据质量[：:]\s*(\d+)/i,
      /论据[：:]\s*(\d+)/i,
      /证据[：:]\s*(\d+)/i,
      /\*\*论据[^*]*\*\*[：:]\s*(\d+)/i,
    ],
    logic: [
      /逻辑[^：:]*[：:]\s*(\d+)/i,
      /\*\*逻辑[^*]*\*\*[：:]\s*(\d+)/i,
    ],
    data: [
      /数据[^：:]*[：:]\s*(\d+)/i,
      /\*\*数据[^*]*\*\*[：:]\s*(\d+)/i,
    ],
  };
  const result: Scores = { evidence: 0, logic: 0, data: 0 };
  let found = false;

  for (const [key, regexes] of Object.entries(patterns)) {
    for (const re of regexes) {
      const match = markdown.match(re);
      if (match) {
        const val = parseInt(match[1]);
        if (val >= 1 && val <= 10) {
          (result as any)[key] = val;
          found = true;
          break;
        }
      }
    }
  }

  // Also try the generic "N/10" pattern adjacent to label keywords
  if (!found) {
    const genericPattern = /(?:论据|证据|逻辑|数据)\s*[：:]\s*(\d+)\s*\/?\s*(?:10)?/gi;
    const scores: number[] = [];
    let m: RegExpExecArray | null;
    while ((m = genericPattern.exec(markdown)) !== null) {
      scores.push(parseInt(m[1]));
    }
    if (scores.length >= 3) {
      result.evidence = scores[0];
      result.logic = scores[1];
      result.data = scores[2];
      found = true;
    }
  }

  return found ? result : null;
}

export default function ScoreCard({ content }: Props) {
  const scores = extractScores(content);
  if (!scores) return null;

  const items = [
    { label: "论据质量", value: scores.evidence },
    { label: "逻辑严密性", value: scores.logic },
    { label: "数据可靠性", value: scores.data },
  ];

  return (
    <div className="score-card">
      <h4>可信度评分</h4>
      {items.map((item) => (
        <div key={item.label} className="score-item">
          <span className="score-label">{item.label}</span>
          <div className="score-bar-bg">
            <div
              className="score-bar-fill"
              style={{ width: `${(item.value / 10) * 100}%` }}
            />
          </div>
          <span className="score-value">{item.value}/10</span>
        </div>
      ))}
    </div>
  );
}
