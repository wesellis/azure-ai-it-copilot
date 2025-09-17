import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

// Components
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import CommandCenter from './pages/CommandCenter';
import Resources from './pages/Resources';
import Incidents from './pages/Incidents';
import CostAnalysis from './pages/CostAnalysis';
import Compliance from './pages/Compliance';
import Predictions from './pages/Predictions';
import Settings from './pages/Settings';
import Login from './pages/Login';

// Context
import { AuthProvider, useAuth } from './context/AuthContext';
import { WebSocketProvider } from './context/WebSocketContext';

// Theme configuration
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#0078D4', // Azure blue
    },
    secondary: {
      main: '#40E0D0', // Turquoise
    },
    background: {
      default: '#1e1e1e',
      paper: '#2d2d2d',
    },
  },
  typography: {
    fontFamily: '"Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#0078D4',
    },
    secondary: {
      main: '#40E0D0',
    },
  },
  typography: {
    fontFamily: '"Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function PrivateRoute({ children }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/login" />;
}

function AppContent() {
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem('darkMode') === 'true'
  );

  useEffect(() => {
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <ThemeProvider theme={darkMode ? darkTheme : lightTheme}>
      <CssBaseline />
      <WebSocketProvider>
        <Router>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route
              path="/*"
              element={
                <PrivateRoute>
                  <Layout darkMode={darkMode} toggleDarkMode={toggleDarkMode}>
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/command" element={<CommandCenter />} />
                      <Route path="/resources" element={<Resources />} />
                      <Route path="/incidents" element={<Incidents />} />
                      <Route path="/cost" element={<CostAnalysis />} />
                      <Route path="/compliance" element={<Compliance />} />
                      <Route path="/predictions" element={<Predictions />} />
                      <Route path="/settings" element={<Settings />} />
                    </Routes>
                  </Layout>
                </PrivateRoute>
              }
            />
          </Routes>
        </Router>
      </WebSocketProvider>
    </ThemeProvider>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;