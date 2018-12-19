provider "aws" {
  region     = "ap-northeast-2"
}

resource "aws_instance" "example" {
  ami = "ami-2a9c4f44"
  instance_type = "t2.micro"
}
