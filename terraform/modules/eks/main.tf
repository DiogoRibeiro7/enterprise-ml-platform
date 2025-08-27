variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the EKS cluster"
  type        = list(string)
}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "mlp-cluster"
  cluster_version = "1.29"
  vpc_id          = var.vpc_id
  subnet_ids      = var.subnet_ids
}

output "cluster_name" {
  value = module.eks.cluster_name
}
