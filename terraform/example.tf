provider "aws" {
  region     = "ap-northeast-2"
}

resource "aws_instance" "example" {
  ami = "ami-f9934097"
  instance_type = "t2.micro"

  provisioner "local-exec" {
    command = "echo ${aws_instance.example.public_ip} > ip_address.txt"
  }
  provisioner "local-exec" {
    when = "destroy"
    command = "rm ip_address.txt"
  }
}

resource "aws_eip" "ip" {
  instance = "${aws_instance.example.id}"
}

