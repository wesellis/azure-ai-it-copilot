import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      // Clear token and redirect to login
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }

    return Promise.reject(error);
  }
);

export default api;

// API service functions
export const apiService = {
  // Authentication
  auth: {
    login: (username, password) => 
      api.post('/auth/login', { username, password }),
    validate: () => 
      api.get('/api/v1/auth/validate'),
  },

  // Commands
  commands: {
    execute: (command, options = {}) =>
      api.post('/api/v1/command', {
        command,
        ...options,
      }),
    getResult: (requestId) =>
      api.get(`/api/v1/command/${requestId}`),
    getHistory: (limit = 50, offset = 0) =>
      api.get('/api/v1/history', { params: { limit, offset } }),
  },

  // Resources
  resources: {
    query: (params) =>
      api.post('/api/v1/resources/query', params),
    create: (resourceType, specifications) =>
      api.post('/api/v1/resources/create', {
        resource_type: resourceType,
        specifications,
      }),
    delete: (resourceId, force = false) =>
      api.delete(`/api/v1/resources/${resourceId}`, {
        params: { force },
      }),
    getDetails: (resourceId) =>
      api.get(`/api/v1/resources/${resourceId}`),
    getDistribution: () =>
      api.get('/api/v1/resources/distribution'),
  },

  // Incidents
  incidents: {
    report: (incident) =>
      api.post('/api/v1/incidents/report', incident),
    getActive: () =>
      api.get('/api/v1/incidents/active'),
    getRecent: () =>
      api.get('/api/v1/incidents/recent'),
  },

  // Cost Analysis
  cost: {
    analyze: (params) =>
      api.post('/api/v1/cost/analyze', params),
    optimize: (optimizationIds, autoApply = false) =>
      api.post('/api/v1/cost/optimize', {
        optimization_ids: optimizationIds,
        auto_apply: autoApply,
      }),
    getTrend: () =>
      api.get('/api/v1/cost/trend'),
  },

  // Compliance
  compliance: {
    check: (frameworks) =>
      api.post('/api/v1/compliance/check', { frameworks }),
    getReport: () =>
      api.get('/api/v1/compliance/report'),
  },

  // Predictions
  predictions: {
    get: (timeHorizon = '7d') =>
      api.get('/api/v1/predictions', { params: { time_horizon: timeHorizon } }),
    getSummary: () =>
      api.get('/api/v1/predictions/summary'),
  },

  // Analytics
  analytics: {
    getUsage: (timeframe = '7d') =>
      api.get('/api/v1/analytics/usage', { params: { timeframe } }),
  },

  // Metrics
  metrics: {
    getSummary: () =>
      api.get('/api/v1/metrics/summary'),
  },

  // Health
  health: {
    check: () =>
      api.get('/health'),
    detailed: () =>
      api.get('/health/detailed'),
  },
};