import { useEffect, useState } from 'react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

type PredictionLabel = 'authentic' | 'human_imposter' | 'ai_generated' | 'unavailable';

type StylometryImportance = {
  feature: string;
  importance: number;
};

type PredictionResult = {
  label: PredictionLabel;
  confidence: number;
  author_similarity: number;
  ai_likelihood: number;
  stylometry: Record<string, number>;
  class_probabilities: Record<string, number>;
  xai_tokens?: { token: string; attention: number }[];
  xai_stylometry?: StylometryImportance[];
  explanation: string;
};

type Article = {
  id: number;
  author: string;
  title: string;
  date: string;
  category: string;
};

function App() {
  const [activeTab, setActiveTab] = useState<'analyze' | 'dataset'>('analyze');
  const [textInput, setTextInput] = useState('');
  const [authorInput, setAuthorInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');
  const [articles, setArticles] = useState<Article[]>([]);
  const [articlesLoading, setArticlesLoading] = useState(true);
  const [articlesError, setArticlesError] = useState('');

  useEffect(() => {
    fetch(`${API_BASE_URL}/articles?limit=10`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Dataset unavailable')))
      .then(setArticles)
      .catch((err) => {
        setArticles([]);
        setArticlesError(err instanceof Error ? err.message : 'Dataset unavailable');
      })
      .finally(() => setArticlesLoading(false));
  }, []);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput || !authorInput) return;
    
    setIsAnalyzing(true);
    setResult(null);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ claimed_author: authorInput, text: textInput }),
      });

      if (!response.ok) {
        const details = await response.json().catch(() => null);
        throw new Error(details?.detail?.[0]?.msg ?? details?.detail ?? 'Analysis failed');
      }

      setResult(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to analyze this text.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const statusClass = result?.label === 'human_imposter' ? 'imposter' : result?.label === 'ai_generated' ? 'ai' : result?.label;
  const confidencePercent = result ? `${(result.confidence * 100).toFixed(1)}%` : '0%';
  const similarityPercent = result ? `${(result.author_similarity * 100).toFixed(1)}%` : '0%';
  const aiPercent = result ? `${(result.ai_likelihood * 100).toFixed(1)}%` : '0%';

  // Get top 3 class probabilities for the detail view
  const topClasses = result?.class_probabilities
    ? Object.entries(result.class_probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 4)
    : [];

  const renderHighlightedText = () => {
    if (!textInput || !result?.xai_tokens) return null;

    const attMap = new Map<string, number>();
    let maxAtt = 0.0001;
    result.xai_tokens.forEach(t => {
      const cleanTok = t.token.replace('##', '').toLowerCase().trim();
      if (cleanTok && cleanTok.length > 1) {
        attMap.set(cleanTok, t.attention);
        if (t.attention > maxAtt) maxAtt = t.attention;
      }
    });

    const words = textInput.split(/(\s+)/);

    return (
      <div className="highlighted-text-box">
        {words.map((w, idx) => {
          const cleanW = w.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"']/g, "").trim();
          let highlightColor = 'transparent';
          let weight = 0;

          if (cleanW) {
            for (const [cleanTok, att] of attMap.entries()) {
              if (cleanW.includes(cleanTok) || cleanTok.includes(cleanW)) {
                weight = att;
                const normWeight = Math.min(1, weight / maxAtt);
                highlightColor = `rgba(255, 140, 0, ${0.15 + normWeight * 0.45})`;
                break;
              }
            }
          }

          if (highlightColor !== 'transparent') {
            return (
              <span
                key={idx}
                className="highlight-word-pill"
                style={{ backgroundColor: highlightColor }}
                title={`Attention weight: ${(weight * 100).toFixed(2)}%`}
              >
                {w}
              </span>
            );
          }
          return <span key={idx}>{w}</span>;
        })}
      </div>
    );
  };

  return (
    <div className="app-container">
      <nav className="navbar glass-panel">
        <div className="nav-brand">
          <span className="logo-icon">🛡️</span>
          <h1>StyloGuard</h1>
        </div>
        <div className="nav-links">
          <button 
            className={`nav-btn ${activeTab === 'analyze' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyze')}
          >
            Analyzer
          </button>
          <button 
            className={`nav-btn ${activeTab === 'dataset' ? 'active' : ''}`}
            onClick={() => setActiveTab('dataset')}
          >
            Dataset
          </button>
        </div>
      </nav>

      <main className="main-content">
        {activeTab === 'analyze' ? (
          <div className="analyzer-view animate-fade-in">
            <header className="view-header">
              <h2>Ghostwriter & Imposter Detection</h2>
              <p>Analyze Indonesian digital media to verify author authenticity using Feature-Fusion Transformers.</p>
            </header>

            <div className="analysis-grid">
              <div className="input-section glass-panel">
                <form onSubmit={handleAnalyze}>
                  <div className="input-group">
                    <label htmlFor="author">Claimed Author</label>
                    <input 
                      id="author"
                      type="text" 
                      placeholder="e.g., Habib Allbi Ferdian, Ahmad Zaki, etc." 
                      value={authorInput}
                      onChange={(e) => setAuthorInput(e.target.value)}
                      required
                    />
                  </div>
                  <div className="input-group">
                    <label htmlFor="article">Article Content</label>
                    <textarea 
                      id="article"
                      placeholder="Paste the article text here for stylometric analysis..." 
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      required
                    />
                  </div>
                  <button 
                    type="submit" 
                    className={`analyze-btn ${isAnalyzing ? 'analyzing' : ''}`}
                    disabled={isAnalyzing || !textInput || !authorInput}
                  >
                    {isAnalyzing ? (
                      <><span className="spinner"></span> Analyzing Stylometrics...</>
                    ) : (
                      'Run Analysis'
                    )}
                  </button>
                </form>
              </div>

              <div className="results-section glass-panel">
                {error ? (
                  <div className="empty-state error-state">
                    <div className="empty-icon">⚠️</div>
                    <h3>Analysis Failed</h3>
                    <p>{error}</p>
                  </div>
                ) : result ? (
                  <div className="result-display animate-fade-in">
                    <h3>Analysis Complete</h3>
                    
                    <div className={`status-card ${statusClass}`}>
                      <div className="status-icon">
                        {result.label === 'authentic' && '✅'}
                        {result.label === 'human_imposter' && '🕵️'}
                        {result.label === 'ai_generated' && '🤖'}
                        {result.label === 'unavailable' && '❔'}
                      </div>
                      <div className="status-text">
                        <h4>
                          {result.label === 'authentic' && 'Authentic Author'}
                          {result.label === 'human_imposter' && 'Human Imposter Detected'}
                          {result.label === 'ai_generated' && 'AI Ghostwriter Detected'}
                          {result.label === 'unavailable' && 'Model Not Loaded'}
                        </h4>
                        <p>{result.explanation}</p>
                      </div>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-box">
                        <span className="metric-label">Model Confidence</span>
                        <span className="metric-value">{confidencePercent}</span>
                        <div className="progress-bar"><div className="fill" style={{width: confidencePercent}}></div></div>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Author Similarity</span>
                        <span className="metric-value">{similarityPercent}</span>
                        <div className="progress-bar">
                          <div className={`fill ${result.author_similarity < 0.5 ? 'warning' : ''}`} style={{width: similarityPercent}}></div>
                        </div>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">AI-Like Signal</span>
                        <span className="metric-value">{aiPercent}</span>
                        <div className="progress-bar"><div className="fill warning" style={{width: aiPercent}}></div></div>
                      </div>
                    </div>

                    {topClasses.length > 0 && (
                      <div className="class-breakdown">
                        <h4 className="breakdown-title">Class Probabilities</h4>
                        {topClasses.map(([label, prob]) => (
                          <div key={label} className="class-row">
                            <span className="class-label">{label}</span>
                            <div className="class-bar-container">
                              <div
                                className={`class-bar ${label === 'AI' ? 'class-bar-ai' : ''}`}
                                style={{ width: `${(prob * 100).toFixed(1)}%` }}
                              ></div>
                            </div>
                            <span className="class-prob">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {result.xai_tokens && result.xai_tokens.length > 0 && (
                      <div className="xai-breakdown">
                        <h4 className="breakdown-title">Explainable AI (xAI) Insights</h4>
                        
                        <p className="xai-subtitle">Semantic Heatmap (Attention-driven context highlighting):</p>
                        {renderHighlightedText()}
                        
                        <p className="xai-subtitle">Top contextual words driving the decision:</p>
                        <div className="xai-tokens-container">
                          {result.xai_tokens.map((xt, idx) => (
                            <div key={idx} className="xai-token-pill">
                              <span className="xai-word">{xt.token.replace('##', '')}</span>
                              <span className="xai-weight">{(xt.attention * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {result.xai_stylometry && result.xai_stylometry.length > 0 && (
                      <div className="stylometry-xai-section animate-fade-in">
                        <h4 className="breakdown-title">Top Stylometric Drivers</h4>
                        <p className="xai-subtitle">Attribution scores mapping style influence (Autograd Gradients):</p>
                        <div className="stylometry-bars-container">
                          {result.xai_stylometry.slice(0, 5).map((item) => {
                            const isPositive = item.importance >= 0;
                            const absVal = Math.abs(item.importance);
                            const maxVal = Math.max(...(result.xai_stylometry?.map(x => Math.abs(x.importance)) || [0.0001]));
                            const percent = Math.min(100, (absVal / maxVal) * 100);
                            
                            const displayName = item.feature
                              .replace('fw_', 'Word: ')
                              .replace(/_/g, ' ')
                              .replace(/\b\w/g, c => c.toUpperCase());
                              
                            return (
                              <div key={item.feature} className="stylo-bar-row">
                                <span className="stylo-bar-label" title={item.feature}>{displayName}</span>
                                <div className="stylo-bar-track">
                                  <div 
                                    className={`stylo-bar-fill ${isPositive ? 'positive' : 'negative'}`}
                                    style={{ width: `${percent}%` }}
                                  ></div>
                                </div>
                                <span className={`stylo-bar-value ${isPositive ? 'positive-text' : 'negative-text'}`}>
                                  {isPositive ? '+' : ''}{item.importance.toFixed(4)}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="empty-state">
                    <div className="empty-icon">📊</div>
                    <h3>Awaiting Input</h3>
                    <p>Enter the article text and author name to begin the stylometric extraction and transformer fusion analysis.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="dataset-view animate-fade-in glass-panel">
            <header className="view-header">
              <h2>Dataset Explorer</h2>
              <p>Sample training data used for the Feature-Fusion Transformer.</p>
            </header>
            
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Author</th>
                    <th>Category</th>
                    <th>Title</th>
                    <th>Date</th>
                    <th>Status</th>
                  </tr>
                </thead>
                  <tbody>
                    {articlesLoading ? (
                      <tr>
                        <td colSpan={5}>Loading articles...</td>
                      </tr>
                    ) : articlesError ? (
                      <tr>
                        <td colSpan={5}>Error: {articlesError}</td>
                      </tr>
                    ) : articles.length ? articles.map((article) => (
                      <tr key={article.id}>
                        <td>{article.author}</td>
                        <td><span className="badge">{article.category}</span></td>
                        <td className="truncate">{article.title}</td>
                        <td>{article.date}</td>
                        <td><span className="badge success">Indexed</span></td>
                      </tr>
                    )) : (
                      <tr>
                        <td colSpan={5}>No backend data loaded yet. Run the backend seed script to populate articles.</td>
                      </tr>
                    )}
                  </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
