variable "vpc_id" {
  type = string
}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "mlp-cluster"
  cluster_version = "1.29"
  vpc_id          = var.vpc_id
  subnet_ids      = []
}

output "cluster_name" {
  value = module.eks.cluster_name
}
