variable "REGION" {
    type = string
    default = "us-central1"
    description = "Region where resources will be located"
}

variable "PROJECT_ID" {
    type = string
    description = "The ID of the project in which the resource belongs"
}

variable "APPLICATION_ID" {
    default = "chatbot"
    type = string
    description = "Application Identifier"
}

variable "ENVIRONMENT" {
    default = "dev"
    type = string
    description = "Environment where resources are provisioned"

    validation {
      condition = contains(["dev", "uat", "staging", "prod"], var.ENVIRONMENT)
      error_message = "Allowed values for ENVIRONMENT are [dev, uat, staging, prod]"
    }
}

variable "TECHNICAL_OWNER" {
    default = "team_pucp"
    type = string
    description = "Team responsable for application support"
}

variable "PROVISIONER" {
    default = "terraform"
    type = string
    description = "Tool used to provision resources"  
}

variable "DATA_CLASSIFICATION" {
    default = "restricted"
    type = string
    description = "A classification for the type of data managed by resources"
}