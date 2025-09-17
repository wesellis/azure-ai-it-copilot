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
  Tooltip,
} from '@mui/material';
import {
  Refresh,
  Delete,
  Edit,
  Add,
  FilterList,
  Cloud,
  Storage,
  Computer,
  NetworkWifi,
} from '@mui/icons-material';
import { apiService } from '../services/api';

function Resources() {
  const [resources, setResources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newResource, setNewResource] = useState({
    type: 'vm',
    name: '',
    location: 'eastus',
    size: 'Standard_B2s',
  });

  const resourceTypes = [
    { value: 'all', label: 'All Resources', icon: <Cloud /> },
    { value: 'vm', label: 'Virtual Machines', icon: <Computer /> },
    { value: 'storage', label: 'Storage Accounts', icon: <Storage /> },
    { value: 'network', label: 'Virtual Networks', icon: <NetworkWifi /> },
  ];

  useEffect(() => {
    fetchResources();
  }, [filterType]);

  const fetchResources = async () => {
    try {
      setLoading(true);
      const response = await apiService.resources.query({
        resource_type: filterType === 'all' ? null : filterType,
      });

      // Mock data for demo
      const mockResources = [
        {
          id: '1',
          name: 'vm-prod-web-01',
          type: 'Microsoft.Compute/virtualMachines',
          location: 'East US',
          resourceGroup: 'rg-production',
          status: 'Running',
          tags: { environment: 'production', team: 'frontend' }
        },
        {
          id: '2',
          name: 'storage-prod-data',
          type: 'Microsoft.Storage/storageAccounts',
          location: 'East US',
          resourceGroup: 'rg-production',
          status: 'Available',
          tags: { environment: 'production', team: 'data' }
        },
        {
          id: '3',
          name: 'vnet-prod-main',
          type: 'Microsoft.Network/virtualNetworks',
          location: 'East US',
          resourceGroup: 'rg-production',
          status: 'Available',
          tags: { environment: 'production' }
        }
      ];

      setResources(mockResources);
    } catch (error) {
      setError('Failed to fetch resources');
      console.error('Error fetching resources:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateResource = async () => {
    try {
      await apiService.resources.create(newResource.type, {
        name: newResource.name,
        location: newResource.location,
        size: newResource.size,
      });
      setCreateDialogOpen(false);
      setNewResource({ type: 'vm', name: '', location: 'eastus', size: 'Standard_B2s' });
      fetchResources();
    } catch (error) {
      setError('Failed to create resource');
    }
  };

  const handleDeleteResource = async (resourceId) => {
    if (!window.confirm('Are you sure you want to delete this resource?')) return;

    try {
      await apiService.resources.delete(resourceId);
      fetchResources();
    } catch (error) {
      setError('Failed to delete resource');
    }
  };

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'running': return 'success';
      case 'stopped': return 'error';
      case 'available': return 'success';
      default: return 'warning';
    }
  };

  const getResourceIcon = (type) => {
    if (type.includes('virtualMachines')) return <Computer />;
    if (type.includes('storageAccounts')) return <Storage />;
    if (type.includes('virtualNetworks')) return <NetworkWifi />;
    return <Cloud />;
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Azure Resources</Typography>
        <Box>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setCreateDialogOpen(true)}
            sx={{ mr: 1 }}
          >
            Create Resource
          </Button>
          <IconButton onClick={fetchResources} disabled={loading}>
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
        {resourceTypes.map((type) => (
          <Grid item xs={12} sm={6} md={3} key={type.value}>
            <Card
              sx={{
                cursor: 'pointer',
                border: filterType === type.value ? 2 : 0,
                borderColor: 'primary.main'
              }}
              onClick={() => setFilterType(type.value)}
            >
              <CardContent>
                <Box display="flex" alignItems="center">
                  {type.icon}
                  <Box ml={2}>
                    <Typography variant="h6">{type.label}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {type.value === 'all' ? resources.length :
                       resources.filter(r => r.type.toLowerCase().includes(type.value)).length} items
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Location</TableCell>
                <TableCell>Resource Group</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Tags</TableCell>
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
              ) : resources.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} align="center">
                    No resources found
                  </TableCell>
                </TableRow>
              ) : (
                resources
                  .filter(resource =>
                    filterType === 'all' || resource.type.toLowerCase().includes(filterType)
                  )
                  .map((resource) => (
                    <TableRow key={resource.id} hover>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          {getResourceIcon(resource.type)}
                          <Typography variant="body2" ml={1}>
                            {resource.name}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {resource.type.split('/')[1]}
                        </Typography>
                      </TableCell>
                      <TableCell>{resource.location}</TableCell>
                      <TableCell>{resource.resourceGroup}</TableCell>
                      <TableCell>
                        <Chip
                          label={resource.status}
                          color={getStatusColor(resource.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box display="flex" gap={0.5} flexWrap="wrap">
                          {Object.entries(resource.tags || {}).map(([key, value]) => (
                            <Chip
                              key={key}
                              label={`${key}: ${value}`}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Tooltip title="Edit">
                          <IconButton size="small">
                            <Edit />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDeleteResource(resource.id)}
                          >
                            <Delete />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Create Resource Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Resource</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                select
                fullWidth
                label="Resource Type"
                value={newResource.type}
                onChange={(e) => setNewResource({ ...newResource, type: e.target.value })}
              >
                <MenuItem value="vm">Virtual Machine</MenuItem>
                <MenuItem value="storage">Storage Account</MenuItem>
                <MenuItem value="network">Virtual Network</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Resource Name"
                value={newResource.name}
                onChange={(e) => setNewResource({ ...newResource, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                select
                fullWidth
                label="Location"
                value={newResource.location}
                onChange={(e) => setNewResource({ ...newResource, location: e.target.value })}
              >
                <MenuItem value="eastus">East US</MenuItem>
                <MenuItem value="westus">West US</MenuItem>
                <MenuItem value="centralus">Central US</MenuItem>
              </TextField>
            </Grid>
            {newResource.type === 'vm' && (
              <Grid item xs={12} sm={6}>
                <TextField
                  select
                  fullWidth
                  label="VM Size"
                  value={newResource.size}
                  onChange={(e) => setNewResource({ ...newResource, size: e.target.value })}
                >
                  <MenuItem value="Standard_B2s">Standard_B2s (2 vCPU, 4GB)</MenuItem>
                  <MenuItem value="Standard_D2s_v3">Standard_D2s_v3 (2 vCPU, 8GB)</MenuItem>
                  <MenuItem value="Standard_D4s_v3">Standard_D4s_v3 (4 vCPU, 16GB)</MenuItem>
                </TextField>
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreateResource}
            variant="contained"
            disabled={!newResource.name}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Resources;