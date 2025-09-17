variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "aicopilot"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "openai_location" {
  description = "Azure region for OpenAI (limited availability)"
  type        = string
  default     = "eastus"
}

variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.28.3"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "Azure AI IT Copilot"
    ManagedBy   = "Terraform"
    Repository  = "https://github.com/wesellis/azure-ai-it-copilot"
  }
}