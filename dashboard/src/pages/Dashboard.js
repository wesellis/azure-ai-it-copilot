import React, { useEffect, useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Box,
  LinearProgress,
  Chip,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Cloud,
  Warning,
  CheckCircle,
  Error,
  AttachMoney,
  Security,
  Storage,
  Computer,
  Refresh,
  MoreVert,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { format } from 'date-fns';
import { useWebSocket } from '../context/WebSocketContext';
import api from '../services/api';

// Chart colors
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

function MetricCard({ title, value, change, icon, color, loading }) {
  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="body2">
              {title}
            </Typography>
            {loading ? (
              <Skeleton variant="text" width={100} height={40} />
            ) : (
              <Typography variant="h4" component="div">
                {value}
              </Typography>
            )}
            {change && (
              <Box display="flex" alignItems="center" mt={1}>
                {change > 0 ? (
                  <TrendingUp color="success" fontSize="small" />
                ) : (
                  <TrendingDown color="error" fontSize="small" />
                )}
                <Typography
                  variant="body2"
                  color={change > 0 ? 'success.main' : 'error.main'}
                  ml={0.5}
                >
                  {Math.abs(change)}%
                </Typography>
              </Box>
            )}
          </Box>
          <Avatar sx={{ bgcolor: color, width: 56, height: 56 }}>
            {icon}
          </Avatar>
        </Box>
      </CardContent>
    </Card>
  );
}

function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState({
    totalResources: 0,
    activeIncidents: 0,
    monthlyCost: 0,
    complianceScore: 0,
  });
  const [resourceData, setResourceData] = useState([]);
  const [costTrend, setCostTrend] = useState([]);
  const [incidentList, setIncidentList] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const { notifications } = useWebSocket();

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Fetch metrics
      const metricsResponse = await api.get('/api/v1/metrics/summary');
      setMetrics(metricsResponse.data);

      // Fetch resource distribution
      const resourcesResponse = await api.get('/api/v1/resources/distribution');
      setResourceData(resourcesResponse.data);

      // Fetch cost trend
      const costResponse = await api.get('/api/v1/cost/trend');
      setCostTrend(costResponse.data);

      // Fetch recent incidents
      const incidentsResponse = await api.get('/api/v1/incidents/recent');
      setIncidentList(incidentsResponse.data);

      // Fetch predictions
      const predictionsResponse = await api.get('/api/v1/predictions/summary');
      setPredictions(predictionsResponse.data);

    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Sample data for charts (in production, this comes from API)
  const performanceData = [
    { time: '00:00', cpu: 45, memory: 62 },
    { time: '04:00', cpu: 38, memory: 58 },
    { time: '08:00', cpu: 72, memory: 75 },
    { time: '12:00', cpu: 85, memory: 82 },
    { time: '16:00', cpu: 78, memory: 80 },
    { time: '20:00', cpu: 62, memory: 68 },
    { time: '24:00', cpu: 48, memory: 60 },
  ];

  const resourceDistribution = [
    { name: 'Virtual Machines', value: 45 },
    { name: 'Storage', value: 25 },
    { name: 'Databases', value: 15 },
    { name: 'Networking', value: 10 },
    { name: 'Other', value: 5 },
  ];

  const costByService = [
    { service: 'Compute', cost: 4500 },
    { service: 'Storage', cost: 2300 },
    { service: 'Database', cost: 3200 },
    { service: 'Network', cost: 1800 },
    { service: 'Other', cost: 1200 },
  ];

  const recentIncidents = [
    {
      id: 1,
      title: 'High CPU Usage on vm-prod-01',
      severity: 'high',
      time: '10 minutes ago',
      status: 'investigating',
    },
    {
      id: 2,
      title: 'Storage Account Reaching Capacity',
      severity: 'medium',
      time: '1 hour ago',
      status: 'resolved',
    },
    {
      id: 3,
      title: 'Network Latency Spike',
      severity: 'low',
      time: '3 hours ago',
      status: 'monitoring',
    },
  ];

  const upcomingPredictions = [
    {
      type: 'failure',
      resource: 'vm-db-02',
      probability: 0.78,
      timeframe: '24-48 hours',
      recommendation: 'Schedule maintenance',
    },
    {
      type: 'capacity',
      resource: 'Storage Account',
      probability: 0.92,
      timeframe: '7 days',
      recommendation: 'Increase storage quota',
    },
  ];

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'resolved':
        return <CheckCircle color="success" />;
      case 'investigating':
        return <Warning color="warning" />;
      case 'monitoring':
        return <Error color="info" />;
      default:
        return <Warning />;
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Dashboard
        </Typography>
        <IconButton onClick={fetchDashboardData} disabled={loading}>
          <Refresh />
        </IconButton>
      </Box>

      {/* Metric Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Resources"
            value={metrics.totalResources || 127}
            change={5}
            icon={<Cloud />}
            color="primary.main"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Incidents"
            value={metrics.activeIncidents || 3}
            change={-25}
            icon={<Warning />}
            color="warning.main"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Monthly Cost"
            value={`$${(metrics.monthlyCost || 13000).toLocaleString()}`}
            change={-8}
            icon={<AttachMoney />}
            color="success.main"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Compliance Score"
            value={`${metrics.complianceScore || 94}%`}
            change={2}
            icon={<Security />}
            color="info.main"
            loading={loading}
          />
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} mb={3}>
        {/* Performance Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              System Performance (24h)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="cpu"
                  stroke="#8884d8"
                  name="CPU %"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="memory"
                  stroke="#82ca9d"
                  name="Memory %"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Resource Distribution */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Resource Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={resourceDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.name}: ${entry.value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {resourceDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Bottom Row */}
      <Grid container spacing={3}>
        {/* Recent Incidents */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent Incidents
            </Typography>
            <List>
              {recentIncidents.map((incident) => (
                <ListItem
                  key={incident.id}
                  secondaryAction={
                    <IconButton edge="end">
                      <MoreVert />
                    </IconButton>
                  }
                >
                  <ListItemAvatar>
                    {getStatusIcon(incident.status)}
                  </ListItemAvatar>
                  <ListItemText
                    primary={incident.title}
                    secondary={
                      <Box>
                        <Chip
                          label={incident.severity}
                          size="small"
                          color={getSeverityColor(incident.severity)}
                          sx={{ mr: 1 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          {incident.time}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Predictions */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Upcoming Predictions
            </Typography>
            {upcomingPredictions.map((prediction, index) => (
              <Alert
                key={index}
                severity={prediction.probability > 0.8 ? 'warning' : 'info'}
                sx={{ mb: 2 }}
              >
                <Typography variant="subtitle2" fontWeight="bold">
                  {prediction.type === 'failure' ? 'Failure Prediction' : 'Capacity Warning'}
                </Typography>
                <Typography variant="body2">
                  {prediction.resource} - {(prediction.probability * 100).toFixed(0)}% probability
                </Typography>
                <Typography variant="caption" display="block">
                  Timeframe: {prediction.timeframe}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Recommendation: {prediction.recommendation}
                </Typography>
              </Alert>
            ))}
          </Paper>
        </Grid>

        {/* Cost by Service */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Cost by Service (Monthly)
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={costByService}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="service" />
                <YAxis />
                <Tooltip formatter={(value) => `$${value}`} />
                <Bar dataKey="cost" fill="#0078D4" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;