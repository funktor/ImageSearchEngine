#!/bin/sh

export AWS_REGION="us-east-1"
export IMAGE_NAME="sagemaker-image-search-repo"
export IMAGE_TAG="latest"
export REGISTRY_ID=$(aws ecr describe-repositories --query 'repositories[?repositoryName == `'$IMAGE_NAME'`].registryId' --output text)
export IMAGE_URI=${REGISTRY_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}

echo "Docker build..."
docker build -t $IMAGE_URI .

echo "ECR login..."
export login=$(aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${REGISTRY_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com)

if [ "$login" == "Login Succeeded" ] ; then
  echo "Login successful..."
  
  echo "Docker push..."
  docker push $IMAGE_URI:$IMAGE_TAG
  
  echo "Done !!!"
else
  exit 0
fi