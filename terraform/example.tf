provider "aws" {
  region     = "ap-northeast-2"
}

resource "aws_s3_bucket" "example" {
  bucket = "terraform-getting-started-guide.worker.yoon"
  acl = "private"
}

resource "aws_instance" "example" {
  ami = "ami-f9934097"
  instance_type = "t2.micro"
  depends_on = ["aws_s3_bucket.example"]
}

resource "aws_eip" "ip" {
  instance = "${aws_instance.example.id}"
}

