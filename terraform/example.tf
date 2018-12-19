provider "aws" {
  region     = "ap-northeast-2"
}

resource "aws_instance" "example" {
  ami = "ami-f9934097"
  instance_type = "t2.micro"
}
