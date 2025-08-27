variable "vpc_id" {
  type = string
}

variable "db_password" {
  description = "The password for the RDS database"
  type        = string
  sensitive   = true
}

resource "aws_db_instance" "this" {
  identifier        = "mlp-db"
  engine            = "postgres"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  username          = "mlp"
  password          = var.db_password
  skip_final_snapshot = true
  vpc_security_group_ids = [aws_security_group.db.id]
}

output "db_endpoint" {
  value = aws_db_instance.this.endpoint
}
