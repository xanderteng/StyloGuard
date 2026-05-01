import { useState } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<'analyze' | 'dataset'>('analyze');
  const [textInput, setTextInput] = useState('');
  const [authorInput, setAuthorInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<null | 'authentic' | 'imposter' | 'ai'>(null);

  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput || !authorInput) return;
    
    setIsAnalyzing(true);
    setResult(null);
    
    // Simulate API call
    setTimeout(() => {
      setIsAnalyzing(false);
      // Random result for mockup
      const outcomes: ('authentic' | 'imposter' | 'ai')[] = ['authentic', 'imposter', 'ai'];
      setResult(outcomes[Math.floor(Math.random() * outcomes.length)]);
    }, 2000);
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
                {result ? (
                  <div className="result-display animate-fade-in">
                    <h3>Analysis Complete</h3>
                    
                    <div className={`status-card ${result}`}>
                      <div className="status-icon">
                        {result === 'authentic' && '✅'}
                        {result === 'imposter' && '🕵️'}
                        {result === 'ai' && '🤖'}
                      </div>
                      <div className="status-text">
                        <h4>
                          {result === 'authentic' && 'Authentic Author'}
                          {result === 'imposter' && 'Human Imposter Detected'}
                          {result === 'ai' && 'AI Ghostwriter Detected'}
                        </h4>
                        <p>
                          {result === 'authentic' && 'The stylometric features match the historical profile of the claimed author.'}
                          {result === 'imposter' && 'Significant deviations in syntactic patterns suggest a different human author.'}
                          {result === 'ai' && 'Low lexical diversity and high predictability indicate LLM generation.'}
                        </p>
                      </div>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-box">
                        <span className="metric-label">IndoBERT Confidence</span>
                        <span className="metric-value">94.2%</span>
                        <div className="progress-bar"><div className="fill" style={{width: '94.2%'}}></div></div>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Lexical Diversity</span>
                        <span className="metric-value">87.5%</span>
                        <div className="progress-bar"><div className="fill" style={{width: '87.5%'}}></div></div>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Syntactic Match</span>
                        <span className="metric-value">
                           {result === 'authentic' ? '92.1%' : '41.3%'}
                        </span>
                        <div className="progress-bar">
                          <div className={`fill ${result === 'authentic' ? '' : 'warning'}`} style={{width: result === 'authentic' ? '92.1%' : '41.3%'}}></div>
                        </div>
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
                  <tr>
                    <td>Matius Alfons Hutajulu</td>
                    <td><span className="badge">berita</span></td>
                    <td className="truncate">Arus Balik Lebaran, Sejumlah Titik Tol Arah Jakarta Padat Malam Ini</td>
                    <td>25 Mar 2026</td>
                    <td><span className="badge success">Verified</span></td>
                  </tr>
                  <tr>
                    <td>Matius Alfons Hutajulu</td>
                    <td><span className="badge">berita</span></td>
                    <td className="truncate">Kemenpar Ungkap 4,2 Juta Warga DKI Tak Mudik...</td>
                    <td>25 Mar 2026</td>
                    <td><span className="badge success">Verified</span></td>
                  </tr>
                  <tr>
                    <td>Unknown</td>
                    <td><span className="badge">opini</span></td>
                    <td className="truncate">Perkembangan Ekonomi Digital di Indonesia 2026</td>
                    <td>24 Mar 2026</td>
                    <td><span className="badge danger">AI Generated</span></td>
                  </tr>
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
