terraform {
    required_providers {
      google = {
        source = "hashicorp/google"
        version = "7.1.1"
      }
    }

    backend "gcs" {
      bucket = "onebank-terraform-state-bucket"
      prefix = "terraform/state"
    }
}

provider "google" {
    project = var.PROJECT_ID
    #region = var.REGION
}

provider "google-beta" {
    project = var.PROJECT_ID
    #region = var.REGION
}