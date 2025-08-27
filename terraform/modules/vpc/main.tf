terraform {
  required_version = ">= 1.4.0"
}

variable "cidr_block" {
  type        = string
  description = "CIDR block for the VPC"
}

resource "aws_vpc" "this" {
  cidr_block = var.cidr_block
  tags = {
    Name = "mlp-vpc"
  }
}

output "vpc_id" {
  value = aws_vpc.this.id
}
