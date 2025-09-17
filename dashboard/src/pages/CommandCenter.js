import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  List,
  ListItem,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Alert,
  IconButton,
  Divider,
  FormControlLabel,
  Switch,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Send as SendIcon,
  Clear as ClearIcon,
  History as HistoryIcon,
  ExpandMore as ExpandMoreIcon,
  ContentCopy as ContentCopyIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  AutoAwesome as AutoAwesomeIcon,
} from '@mui/icons-material';
import { format } from 'date-fns';
import api from '../services/api';
import { useWebSocket } from '../context/WebSocketContext';

const commandExamples = [
  {
    category: 'Resource Management',
    commands: [
      'Create a Linux VM with 8GB RAM in East US',
      'List all VMs in production resource group',
      'Delete unused storage accounts',
      'Show me all resources tagged as development',
    ],
  },
  {
    category: 'Incident Response',
    commands: [
      'Diagnose high CPU usage on vm-prod-001',
      'Check for any critical alerts in the last hour',
      'Investigate slow response times on the API gateway',
      'Auto-remediate memory issues on web servers',
    ],
  },
  {
    category: 'Cost Optimization',
    commands: [
      'Find all idle resources wasting money',
      'Optimize our Azure costs without affecting production',
      'Show me Reserved Instance recommendations',
      'What resources are consuming the most budget?',
    ],
  },
  {
    category: 'Compliance & Security',
    commands: [
      'Check if all storage accounts are encrypted',
      'Run CIS benchmark compliance check',
      'Find resources without proper tags',
      'Audit admin access in the last 7 days',
    ],
  },
  {
    category: 'Predictive Analytics',
    commands: [
      'Predict failures for next 48 hours',
      'When will we run out of storage capacity?',
      'Analyze usage trends and forecast costs',
      'Detect anomalies in resource behavior',
    ],
  },
];

function CommandMessage({ message, isUser }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(
      isUser ? message.command : JSON.stringify(message.result, null, 2)
    );
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
      case 'success':
        return <CheckCircleIcon color="success" />;
      case 'failed':
      case 'error':
        return <ErrorIcon color="error" />;
      case 'processing':
        return <CircularProgress size={20} />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  return (
    <ListItem
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Card
        sx={{
          maxWidth: '80%',
          bgcolor: isUser ? 'primary.dark' : 'background.paper',
        }}
      >
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start">
            <Box flex={1}>
              {isUser ? (
                <>
                  <Typography variant="body1">{message.command}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {format(new Date(message.timestamp), 'HH:mm:ss')}
                  </Typography>
                </>
              ) : (
                <>
                  <Box display="flex" alignItems="center" mb={1}>
                    {getStatusIcon(message.status)}
                    <Typography variant="subtitle2" ml={1}>
                      Status: {message.status}
                    </Typography>
                    {message.execution_time && (
                      <Chip
                        label={`${message.execution_time.toFixed(2)}s`}
                        size="small"
                        sx={{ ml: 1 }}
                      />
                    )}
                  </Box>
                  {message.result && (
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography>View Result</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <pre style={{ overflow: 'auto', fontSize: '0.85rem' }}>
                          {JSON.stringify(message.result, null, 2)}
                        </pre>
                      </AccordionDetails>
                    </Accordion>
                  )}
                </>
              )}
            </Box>
            <IconButton size="small" onClick={handleCopy}>
              {copied ? <CheckCircleIcon fontSize="small" /> : <ContentCopyIcon fontSize="small" />}
            </IconButton>
          </Box>
        </CardContent>
      </Card>
    </ListItem>
  );
}

function CommandCenter() {
  const [command, setCommand] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [autoApprove, setAutoApprove] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  const [history, setHistory] = useState([]);
  const [showExamples, setShowExamples] = useState(true);
  const messagesEndRef = useRef(null);
  const { sendMessage } = useWebSocket();

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadHistory = async () => {
    try {
      const response = await api.get('/api/v1/history?limit=10');
      setHistory(response.data.history);
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!command.trim() || loading) return;

    const userMessage = {
      command: command,
      timestamp: new Date().toISOString(),
      isUser: true,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setCommand('');

    try {
      const response = await api.post('/api/v1/command', {
        command: command,
        auto_approve: autoApprove,
        dry_run: dryRun,
        context: {
          source: 'command_center',
        },
      });

      const resultMessage = {
        ...response.data,
        isUser: false,
      };

      setMessages((prev) => [...prev, resultMessage]);

      // Send WebSocket message for real-time updates
      sendMessage({
        type: 'command_executed',
        data: {
          command: command,
          result: response.data,
        },
      });

      // Reload history
      loadHistory();
    } catch (error) {
      const errorMessage = {
        status: 'error',
        result: {
          error: error.response?.data?.detail || 'Command execution failed',
        },
        isUser: false,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleCommand) => {
    setCommand(exampleCommand);
    setShowExamples(false);
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  return (
    <Box sx={{ height: 'calc(100vh - 100px)', display: 'flex', flexDirection: 'column' }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4" component="h1">
          Command Center
        </Typography>
        <Box>
          <FormControlLabel
            control={
              <Switch
                checked={autoApprove}
                onChange={(e) => setAutoApprove(e.target.checked)}
                size="small"
              />
            }
            label="Auto-Approve"
          />
          <FormControlLabel
            control={
              <Switch
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
                size="small"
              />
            }
            label="Dry Run"
          />
          <Tooltip title="Clear chat">
            <IconButton onClick={handleClearChat} disabled={messages.length === 0}>
              <ClearIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {showExamples && messages.length === 0 && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            <AutoAwesomeIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Try these example commands:
          </Typography>
          {commandExamples.map((category) => (
            <Box key={category.category} mb={2}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                {category.category}
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {category.commands.map((cmd) => (
                  <Chip
                    key={cmd}
                    label={cmd}
                    onClick={() => handleExampleClick(cmd)}
                    clickable
                    variant="outlined"
                    size="small"
                  />
                ))}
              </Box>
            </Box>
          ))}
        </Paper>
      )}

      <Paper sx={{ flex: 1, overflow: 'auto', p: 2, mb: 2 }}>
        {messages.length === 0 ? (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            height="100%"
          >
            <Typography variant="h6" color="text.secondary">
              Start by typing a natural language command below
            </Typography>
            <Typography variant="body2" color="text.secondary" mt={1}>
              I can help you manage Azure resources, diagnose issues, optimize costs, and more!
            </Typography>
          </Box>
        ) : (
          <List>
            {messages.map((msg, index) => (
              <CommandMessage key={index} message={msg} isUser={msg.isUser} />
            ))}
            <div ref={messagesEndRef} />
          </List>
        )}
      </Paper>

      <Paper component="form" onSubmit={handleSubmit} sx={{ p: 2 }}>
        <Box display="flex" gap={2}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your command here... (e.g., 'Create a VM with 16GB RAM' or 'Find idle resources')"  
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            disabled={loading}
            InputProps={{
              endAdornment: loading && <CircularProgress size={20} />,
            }}
          />
          <Button
            type="submit"
            variant="contained"
            endIcon={<SendIcon />}
            disabled={!command.trim() || loading}
            sx={{ minWidth: 120 }}
          >
            {loading ? 'Processing...' : 'Send'}
          </Button>
        </Box>
        {dryRun && (
          <Alert severity="info" sx={{ mt: 1 }}>
            Dry run mode is enabled. Commands will be simulated without making actual changes.
          </Alert>
        )}
      </Paper>
    </Box>
  );
}

export default CommandCenter;