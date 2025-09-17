import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  Grid,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
} from '@mui/lab';
import {
  Error,
  Warning,
  Info,
  CheckCircle,
  PlayArrow,
  Stop,
  Refresh,
  Add,
  ExpandMore,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { apiService } from '../services/api';

function Incidents() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);
  const [newIncident, setNewIncident] = useState({
    description: '',
    severity: 'medium',
    affectedResources: '',
    symptoms: '',
  });

  useEffect(() => {
    fetchIncidents();
  }, []);

  const fetchIncidents = async () => {
    try {
      setLoading(true);

      // Mock data for demo
      const mockIncidents = [
        {
          id: '1',
          title: 'High CPU Usage on vm-prod-web-01',
          description: 'CPU utilization consistently above 90% for the past 30 minutes',
          severity: 'high',
          status: 'investigating',
          affectedResources: ['vm-prod-web-01'],
          symptoms: ['High CPU', 'Slow response times'],
          created: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
          updated: new Date(Date.now() - 30 * 60 * 1000), // 30 mins ago
          timeline: [
            { time: new Date(Date.now() - 2 * 60 * 60 * 1000), action: 'Incident created', status: 'created' },
            { time: new Date(Date.now() - 90 * 60 * 1000), action: 'Started investigation', status: 'investigating' },
            { time: new Date(Date.now() - 30 * 60 * 1000), action: 'Root cause identified: Memory leak', status: 'investigating' },
          ]
        },
        {
          id: '2',
          title: 'Storage Account Connectivity Issues',
          description: 'Intermittent connection failures to storage-prod-data',
          severity: 'medium',
          status: 'resolved',
          affectedResources: ['storage-prod-data'],
          symptoms: ['Connection timeouts', 'Failed uploads'],
          created: new Date(Date.now() - 24 * 60 * 60 * 1000), // 24 hours ago
          updated: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
          timeline: [
            { time: new Date(Date.now() - 24 * 60 * 60 * 1000), action: 'Incident created', status: 'created' },
            { time: new Date(Date.now() - 20 * 60 * 60 * 1000), action: 'Escalated to network team', status: 'investigating' },
            { time: new Date(Date.now() - 2 * 60 * 60 * 1000), action: 'Fixed network configuration', status: 'resolved' },
          ]
        },
        {
          id: '3',
          title: 'Database Performance Degradation',
          description: 'Query response times increased by 300%',
          severity: 'critical',
          status: 'remediation_in_progress',
          affectedResources: ['sql-prod-main'],
          symptoms: ['Slow queries', 'Lock contention', 'High wait times'],
          created: new Date(Date.now() - 45 * 60 * 1000), // 45 mins ago
          updated: new Date(Date.now() - 5 * 60 * 1000), // 5 mins ago
          timeline: [
            { time: new Date(Date.now() - 45 * 60 * 1000), action: 'Incident created', status: 'created' },
            { time: new Date(Date.now() - 30 * 60 * 1000), action: 'Auto-remediation started', status: 'remediation_in_progress' },
            { time: new Date(Date.now() - 5 * 60 * 1000), action: 'Index rebuild initiated', status: 'remediation_in_progress' },
          ]
        }
      ];

      setIncidents(mockIncidents);
    } catch (error) {
      setError('Failed to fetch incidents');
      console.error('Error fetching incidents:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReportIncident = async () => {
    try {
      await apiService.incidents.report({
        description: newIncident.description,
        severity: newIncident.severity,
        affected_resources: newIncident.affectedResources.split(',').map(r => r.trim()),
        symptoms: newIncident.symptoms.split(',').map(s => s.trim()),
        auto_remediate: true,
      });
      setReportDialogOpen(false);
      setNewIncident({ description: '', severity: 'medium', affectedResources: '', symptoms: '' });
      fetchIncidents();
    } catch (error) {
      setError('Failed to report incident');
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'resolved': return 'success';
      case 'investigating': return 'warning';
      case 'remediation_in_progress': return 'info';
      case 'created': return 'default';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'resolved': return <CheckCircle />;
      case 'investigating': return <Warning />;
      case 'remediation_in_progress': return <PlayArrow />;
      case 'created': return <Info />;
      default: return <Error />;
    }
  };

  const getTimelineIcon = (status) => {
    switch (status) {
      case 'resolved': return <CheckCircle color="success" />;
      case 'investigating': return <Warning color="warning" />;
      case 'remediation_in_progress': return <PlayArrow color="info" />;
      case 'created': return <Info color="action" />;
      default: return <Error color="error" />;
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Incident Management</Typography>
        <Box>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setReportDialogOpen(true)}
            sx={{ mr: 1 }}
          >
            Report Incident
          </Button>
          <IconButton onClick={fetchIncidents} disabled={loading}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Error color="error" />
                <Box ml={2}>
                  <Typography variant="h6">Critical</Typography>
                  <Typography variant="h4" color="error">
                    {incidents.filter(i => i.severity === 'critical').length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Warning color="warning" />
                <Box ml={2}>
                  <Typography variant="h6">High</Typography>
                  <Typography variant="h4" color="warning.main">
                    {incidents.filter(i => i.severity === 'high').length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <PlayArrow color="info" />
                <Box ml={2}>
                  <Typography variant="h6">Active</Typography>
                  <Typography variant="h4" color="info.main">
                    {incidents.filter(i => i.status !== 'resolved').length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <CheckCircle color="success" />
                <Box ml={2}>
                  <Typography variant="h6">Resolved</Typography>
                  <Typography variant="h4" color="success.main">
                    {incidents.filter(i => i.status === 'resolved').length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Incident</TableCell>
                <TableCell>Severity</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Affected Resources</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Last Updated</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={7} align="center">
                    <CircularProgress />
                  </TableCell>
                </TableRow>
              ) : incidents.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} align="center">
                    No incidents found
                  </TableCell>
                </TableRow>
              ) : (
                incidents.map((incident) => (
                  <TableRow key={incident.id} hover>
                    <TableCell>
                      <Box>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {incident.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {incident.description}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={incident.severity}
                        color={getSeverityColor(incident.severity)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        {getStatusIcon(incident.status)}
                        <Chip
                          label={incident.status.replace('_', ' ')}
                          color={getStatusColor(incident.status)}
                          size="small"
                          sx={{ ml: 1 }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={0.5} flexWrap="wrap">
                        {incident.affectedResources.map((resource) => (
                          <Chip
                            key={resource}
                            label={resource}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    </TableCell>
                    <TableCell>
                      {format(incident.created, 'MMM dd, HH:mm')}
                    </TableCell>
                    <TableCell>
                      {format(incident.updated, 'MMM dd, HH:mm')}
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        onClick={() => setSelectedIncident(incident)}
                      >
                        View Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Incident Details Dialog */}
      <Dialog
        open={Boolean(selectedIncident)}
        onClose={() => setSelectedIncident(null)}
        maxWidth="md"
        fullWidth
      >
        {selectedIncident && (
          <>
            <DialogTitle>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="h6">{selectedIncident.title}</Typography>
                <Chip
                  label={selectedIncident.severity}
                  color={getSeverityColor(selectedIncident.severity)}
                />
              </Box>
            </DialogTitle>
            <DialogContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>Description</Typography>
                  <Typography variant="body2" paragraph>
                    {selectedIncident.description}
                  </Typography>

                  <Typography variant="subtitle2" gutterBottom>Symptoms</Typography>
                  <Box display="flex" gap={0.5} flexWrap="wrap" mb={2}>
                    {selectedIncident.symptoms.map((symptom) => (
                      <Chip key={symptom} label={symptom} size="small" />
                    ))}
                  </Box>

                  <Typography variant="subtitle2" gutterBottom>Affected Resources</Typography>
                  <Box display="flex" gap={0.5} flexWrap="wrap">
                    {selectedIncident.affectedResources.map((resource) => (
                      <Chip key={resource} label={resource} variant="outlined" size="small" />
                    ))}
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>Timeline</Typography>
                  <Timeline>
                    {selectedIncident.timeline.map((event, index) => (
                      <TimelineItem key={index}>
                        <TimelineSeparator>
                          <TimelineDot>
                            {getTimelineIcon(event.status)}
                          </TimelineDot>
                          {index < selectedIncident.timeline.length - 1 && <TimelineConnector />}
                        </TimelineSeparator>
                        <TimelineContent>
                          <Typography variant="body2" fontWeight="bold">
                            {event.action}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {format(event.time, 'MMM dd, HH:mm')}
                          </Typography>
                        </TimelineContent>
                      </TimelineItem>
                    ))}
                  </Timeline>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedIncident(null)}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* Report Incident Dialog */}
      <Dialog open={reportDialogOpen} onClose={() => setReportDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Report New Incident</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                value={newIncident.description}
                onChange={(e) => setNewIncident({ ...newIncident, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                select
                fullWidth
                label="Severity"
                value={newIncident.severity}
                onChange={(e) => setNewIncident({ ...newIncident, severity: e.target.value })}
              >
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Affected Resources (comma separated)"
                value={newIncident.affectedResources}
                onChange={(e) => setNewIncident({ ...newIncident, affectedResources: e.target.value })}
                placeholder="vm-prod-01, storage-account-main"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Symptoms (comma separated)"
                value={newIncident.symptoms}
                onChange={(e) => setNewIncident({ ...newIncident, symptoms: e.target.value })}
                placeholder="High CPU, Slow response times, Connection errors"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReportDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleReportIncident}
            variant="contained"
            disabled={!newIncident.description}
          >
            Report Incident
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Incidents;