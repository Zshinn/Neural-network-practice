#include "myNetwork.h"
#include "../../tensor/function/FHeader.h"
#include <bitset>
using std::bitset;
#include<cmath>
using namespace std;

namespace myNetwork
{
// todo 初始化参数
float minmax = 0.01F;
float learningRate = 0.3F;           // learning rate
int nEpoch = 1000;
XTensor one;


void Init(mymodel &model);
void InitGrad(mymodel &model, mymodel &grad);
void fit(TensorList inputList, TensorList goldList, mymodel &model, mynet &net, mymodel &grad,
	float &totalLoss, XTensor &theta1_grad, XTensor &theta2_grad);
void Train(float trainDataX[4][2], float trainDataY[4], int dataSize, mymodel &model);
void Forword(XTensor &input, mymodel &model, mynet &net);
void CrossEntropy_back(XTensor &gold, mymodel modle, mynet net,
	XTensor &dw1, XTensor &dw2);
void Backward(XTensor &input, XTensor &gold, mymodel &model, mymodel &grad, mynet &net);
void CleanGrad(mymodel &grad);
void Test(mymodel &model);
XTensor reshape(int a, int b, XTensor de);
XTensor trans(int a, int b, XTensor de);

int myNetMain(int argc, const char ** argv)
{
	mymodel model;
	model.h_size = 2;
	const int dataSize = 4;
	model.devID = -1;

	float X[4][2] = { {0,0},{0,1},{1,0},{1,1} };
	float Y[4] = { 0,1,1,0 };
	
	InitTensor2D(&model.y, 4, 1, X_FLOAT, model.devID);
	for (int i = 0; i < 4; i++) {
		model.y.Set2D(Y[i], i, 0);
	}

	Init(model);

	Train(X, Y, dataSize, model);
	Test(model);
	return 0;
}

void Init(mymodel &model) {
	InitTensor2D(&model.w1, 2, 3, X_FLOAT, model.devID);
	InitTensor2D(&model.w2, 1,3, X_FLOAT, model.devID);
	model.w1.SetDataRand(-minmax, minmax);
	model.w2.SetDataRand(-minmax, minmax);
	InitTensor2D(&one, 4,1, X_FLOAT, model.devID);
	for (int i = 0; i < 4; i++) {
		one.Set2D(1.0, i,0);
	}

	printf("Init model finished \n");
}

void InitGrad(mymodel &model, mymodel &grad) {
	InitTensor(&grad.w1, &model.w1);
	InitTensor(&grad.w2, &model.w2);

	grad.h_size = model.h_size;
	grad.devID = model.devID;
}

XTensor reshape(int a, int b, XTensor de) {
	XTensor *shape = NewTensor2D(a, b, X_FLOAT, -1);
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < b; j++) {
			shape->Set2D(de.Get(0), i, j);
		}
	}
	return *shape;
}

XTensor trans(int a, int b, XTensor de) {
	XTensor *shape = NewTensor2D(b, a, X_FLOAT, -1);
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < b; j++) {
			shape->Set2D(de.Get2D(i,j),j ,i );
		}
	}
	return *shape;
}

void fit(TensorList inputList, TensorList goldList, mymodel &model, mynet &net, mymodel &grad,
	float &totalLoss, XTensor &theta1_grad, XTensor &theta2_grad) {

	XTensor dedw1;
	XTensor dedw2;
	XTensor * output_all = NewTensor2D(4, 1, X_FLOAT, model.devID);
	InitTensor2D(&dedw1, 2, 3, X_FLOAT, model.devID);
	InitTensor2D(&dedw2, 1, 3, X_FLOAT, model.devID);
	dedw1.SetZeroAll();
	dedw2.SetZeroAll();

	for (int i = 0; i < 4; i++) {
		XTensor *input = inputList.GetItem(i);
		XTensor *gold = goldList.GetItem(i);
		Forword(*input, model, net);
		output_all->Set2D(net.output.Get(0), i, 0);
		//Backward(*input, *gold, model, grad, net);
		CrossEntropy_back(*gold, model, net, grad.w1, grad.w2);
		printf("out of cross \n");
		dedw1 = dedw1 + grad.w1;
		dedw2 = dedw2 + grad.w2;
		printf("循环 ok");
	}
	theta1_grad = (1.0 / 4)*dedw1;
	theta2_grad = (1.0 / 4)*dedw2;
	XTensor t11 = (-1.0)*(model.y);//[4 1]
	XTensor t1 = trans(4, 1, t11);
	
	XTensor t2 = MatrixMul(t1, Log(output_all)); //[1]
	XTensor t33 = one - (model.y); //[4 1]
	XTensor t4 = one - Log(output_all); //[4 1]
	XTensor t3 = trans(4, 1, t33);
	XTensor t5 = MatrixMul(t3, t4);//[1]
	XTensor loss = (1.0 / 4)*(t2 - t5); //[1]
	totalLoss = loss.Get(0);
	printf("fit done \n");

}


void Train(float trainDataX[4][2], float trainDataY[4], int dataSize, mymodel &model) {
	printf("pepare date for train\n");
	TensorList inputList; 
	TensorList goldList;
	XTensor*  inputData; //[3]
	XTensor*  goldData; //[4]

	for (int i = 0; i < 4; i++) {
		inputData = NewTensor2D(3,1, X_FLOAT, model.devID);
		inputData->Set2D(1, 0,0);
		for (int j = 0; j < 2; j++)
		{
			inputData->Set2D(trainDataX[i][j], j+1,0);
			inputList.Add(inputData);
		}
		goldData = NewTensor1D(1, X_FLOAT, model.devID);
		goldData->Set1D(trainDataY[i], 0);
		goldList.Add(goldData);
	}
	
	printf("start train\n");
	mynet net;
	mymodel grad;
	InitGrad(model, grad);
	for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
	{
		XTensor theta1_grad;
		XTensor theta2_grad;
		printf("epoch %d\n", epochIndex);
		float totalLoss = 0;
		fit(inputList, goldList,model,net,grad, totalLoss, theta1_grad, theta2_grad);
		model.w1 = model.w1 - learningRate * theta1_grad;
		model.w2 = model.w2 - learningRate * theta2_grad;
		CleanGrad(grad);
		printf("%f\n", totalLoss);
	}
}

void Forword(XTensor &input, mymodel &model, mynet &net) {
	net.input = input;
	net.hidden_state1 = MatrixMul(model.w1, net.input); //[2 * 1]
	XTensor tmp= Sigmoid(net.hidden_state1);
	// 拼
	XTensor*  a2= NewTensor2D(3,1, X_FLOAT, model.devID);
	a2->Set2D(1, 0,0);
	for (int i = 0; i < 2; i++) {
		a2->Set2D(tmp.Get(i), i + 1,0);
	}
	net.hidden_state2 = Sigmoid(a2);
	net.hidden_state3 = MatrixMul(model.w2, net.hidden_state2);
	//net.output = MatrixMul(net.hidden_state3, model.w2);
	net.output = Sigmoid(net.hidden_state3);
	printf("Forword ok \n");
	
}


void CrossEntropy_back(XTensor &gold,mymodel modle,mynet net, 
						XTensor &dw1, XTensor &dw2)
{
	XTensor delta3 = net.output - gold; //[1]
	XTensor delta_3 = reshape(3,1,delta3);
	XTensor tmp1 = MatrixMul(modle.w2, delta_3); //[1]
	printf("tmp ok \n");

	XTensor * dedy=NewTensor2D(2,1, X_FLOAT, modle.devID);
	XTensor * dedx=NewTensor2D(2,1, X_FLOAT, modle.devID);
	XTensor * y= NewTensor2D(2,1, X_FLOAT, modle.devID);;
	dedy->SetZeroAll();
	dedx->SetZeroAll();
	y->SetZeroAll();
	XTensor * x = &net.hidden_state1;
	_SigmoidBackward(y, x, dedy, dedx);
	XTensor tmp = reshape(2, 2, tmp1);
	XTensor delta_2 = MatrixMul(tmp, dedx); //[2,1]
	dw2 = delta_3* net.hidden_state2;//[1]
	XTensor a1 = net.input;
	XTensor a = reshape(1,3,a1);
	dw1 = MatrixMul(delta_2, a); //[2,1] [1 3]=[2,3]
	printf("CrossEntropy_back all ok \n");
}

void Backward(XTensor &input, XTensor &gold, mymodel &model, mymodel &grad, mynet &net)
{
	CrossEntropy_back(gold,model,net,grad.w1,grad.w2);
	printf("Backward ok \n");
}

void CleanGrad(mymodel &grad)
{
	grad.w1.SetZeroAll();
	grad.w2.SetZeroAll();
}

void Test(mymodel &model)
{
	mynet net;
	bitset<3> bit1(3);
	bitset<3> bit2(5);
	float testData[2][3];
	for (int j = 0; j < 3; j++)
	{
		testData[0][j] = bit1[j];
		testData[1][j] = bit2[j];
	}
	XTensor*  inputData = NewTensor2D(3,1, X_FLOAT, model.devID);
	inputData->Set2D(1, 0,0);
	float ans[3] = {-1,-1,-1};
	for (int i = 0; i < 3; i++) {
		inputData->Set2D(testData[0][i], i + 1,0);
		inputData->Set2D(testData[1][i], i + 2,0);
		Forword(*inputData, model, net);
		ans[i] = net.output.Get(0);
	}
	printf("%f%f%f", ans[0],ans[1],ans[2]);
}

}
