a1 = 12;  % Scale param
b1 = 20;  % Shape param

a2 = 14; 
b2 = 8; 

a3 = 12; 
b3 = 8; 

a4 = 12.5; 
b4 = 19.5; 

a5 = 13.5; 
b5 = 7.5; 

a6 = 13; 
b6 = 7; 

a7 = 13.5; 
b7 = 18; 

a8 = 14.5; 
b8 = 7.5; 


WeibullPDF1 = makedist('Weibull', 'a', a1, 'b', b1);             % Zone 1
WeibullPDF2 = makedist('Weibull', 'a', a2, 'b', b2);             % Zone 2
WeibullPDF3 = makedist('Weibull', 'a', a3, 'b', b3);             % Zone 3
WeibullPDF4 = makedist('Weibull', 'a', a4, 'b', b4);             % Zone 4
WeibullPDF5 = makedist('Weibull', 'a', a5, 'b', b5);             % Zone 5
WeibullPDF6 = makedist('Weibull', 'a', a6, 'b', b6);             % Zone 6
WeibullPDF7 = makedist('Weibull', 'a', a7, 'b', b7);             % Zone 7
WeibullPDF8 = makedist('Weibull', 'a', a8, 'b', b8);             % Zone 8


% % Convert wind speed to wind power
left_truc1 = 0.5;    % Left truncation
right_truc1 = 15;    % Right truncation

left_truc2 = 2;  
right_truc2 = 18; 

left_truc3 = 1;  
right_truc3 = 20;  

left_truc4 = 0.55;  
right_truc4 = 15.3; 

left_truc5 = 2.1;  
right_truc5 = 18.2;  

left_truc6 = 1.1;  
right_truc6 = 19.8; 

left_truc7 = 0.52;  
right_truc7 = 14.9;  

left_truc8 = 2.3;  
right_truc8 = 18.5; 









x = linspace(0, 20, 1000);

pdf1 = WeibullPDF1.pdf(x);
pdf2 = WeibullPDF2.pdf(x);
pdf3 = WeibullPDF3.pdf(x);
pdf4 = WeibullPDF4.pdf(x);
pdf5 = WeibullPDF5.pdf(x);
pdf6 = WeibullPDF6.pdf(x);
pdf7 = WeibullPDF7.pdf(x);
pdf8 = WeibullPDF8.pdf(x);


plot(x, pdf1, LineWidth=2);
hold on;
plot(x, pdf2, LineWidth=2);
hold on;
plot(x, pdf3, LineWidth=2);
hold on;
plot(x, pdf4, LineWidth=2);
hold on;
plot(x, pdf5, LineWidth=2);
hold on;
plot(x, pdf6, LineWidth=2);
hold on;
plot(x, pdf7, LineWidth=2);
hold on;
plot(x, pdf8, LineWidth=2);

legend('pdf1', 'pdf2', 'pdf3', 'pdf4', 'pdf5', 'pdf6', 'pdf7', 'pdf8');


































