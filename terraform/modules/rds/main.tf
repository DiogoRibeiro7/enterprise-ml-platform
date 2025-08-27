variable "vpc_id" {
  type = string
}

resource "aws_db_instance" "this" {
  identifier        = "mlp-db"
  engine            = "postgres"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  username          = "mlp"
  password          = "mlp12345"
  skip_final_snapshot = true
  vpc_security_group_ids = []
}

output "db_endpoint" {
  value = aws_db_instance.this.endpoint
}
