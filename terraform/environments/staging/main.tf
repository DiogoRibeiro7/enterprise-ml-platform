provider "aws" {
  region = var.region
}

module "vpc" {
  source     = "../../modules/vpc"
  cidr_block = "10.1.0.0/16"
}

module "eks" {
  source  = "../../modules/eks"
  vpc_id  = module.vpc.vpc_id
}

module "rds" {
  source = "../../modules/rds"
  vpc_id = module.vpc.vpc_id
}
