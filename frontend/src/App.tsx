import { useEffect, useState } from 'react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
// Keep aligned with the backend threshold used to label strong author matches.
const SIMILARITY_THRESHOLD = 0.86;

type PredictionLabel = 'authentic' | 'human_imposter' | 'ai_generated' | 'uncertain' | 'unknown';

type PredictionResult = {
  label: PredictionLabel;
  confidence: number;
  author_similarity: number;
  ai_likelihood: number;
  known_articles: number;
  stylometry: {
    lexical_diversity: number;
    avg_sentence_length: number;
    punctuation_density: number;
  };
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
  const lexicalPercent = result ? `${(result.stylometry.lexical_diversity * 100).toFixed(1)}%` : '0%';
  const syntacticPercent = result ? `${(result.author_similarity * 100).toFixed(1)}%` : '0%';

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
                      placeholder="e.g., Matius Alfons Hutajulu" 
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
                        {result.label === 'uncertain' && '🧭'}
                        {result.label === 'unknown' && '❔'}
                      </div>
                      <div className="status-text">
                        <h4>
                          {result.label === 'authentic' && 'Authentic Author'}
                          {result.label === 'human_imposter' && 'Human Imposter Detected'}
                          {result.label === 'ai_generated' && 'AI Ghostwriter Detected'}
                          {result.label === 'uncertain' && 'Uncertain Match'}
                          {result.label === 'unknown' && 'Insufficient Author Data'}
                        </h4>
                        <p>{result.explanation}</p>
                        <p className="known-count">Compared against {result.known_articles} known article(s).</p>
                      </div>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-box">
                        <span className="metric-label">Decision Confidence</span>
                        <span className="metric-value">{confidencePercent}</span>
                        <div className="progress-bar"><div className="fill" style={{width: confidencePercent}}></div></div>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Lexical Diversity</span>
                        <span className="metric-value">{lexicalPercent}</span>
                        <div className="progress-bar"><div className="fill" style={{width: lexicalPercent}}></div></div>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Syntactic Match</span>
                        <span className="metric-value">{syntacticPercent}</span>
                        <div className="progress-bar">
                          <div className={`fill ${result.author_similarity >= SIMILARITY_THRESHOLD ? '' : 'warning'}`} style={{width: syntacticPercent}}></div>
                        </div>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">AI-Like Signal</span>
                        <span className="metric-value">{(result.ai_likelihood * 100).toFixed(1)}%</span>
                        <div className="progress-bar"><div className="fill warning" style={{width: `${result.ai_likelihood * 100}%`}}></div></div>
                      </div>
                    </div>
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
