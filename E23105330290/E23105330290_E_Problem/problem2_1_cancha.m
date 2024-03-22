ED_value = xlsread("ED value.xlsx");
ED_time = xlsread("ED time.xlsx");
ED_count = xlsread("ED Count.xlsx");

clear;clc;
ED_count = xlsread("ED Count.xlsx");
ED_value = xlsread("ED value insert.xlsx");
ED_time = xlsread("ED time insert.xlsx");
ED_class = xlsread("gmm聚类.xlsx");

%% 进行数据拟合
for i = 2:length(ED_time)
    a = ED_time(i,:);
    b = ED_value(i,:);
    class = ED_class(i,:);
    show = true;
    [fitresult{i}, gof{i}] = createFit(a, b ,class, i ,show);
end


%% 计算所有参数均值后拟合
cal_num = 200;
max_time = max(max(ED_time));
cal_list = linspace(0, max_time, cal_num);
aver_y = zeros(1,cal_num);
aver_sum = zeros(1,cal_num); %计数
for i = 2:length(ED_time)
    max_time_tmp = max(ED_time(i,:));
    for j = 1:length(cal_list)
        if(max_time_tmp > cal_list(j))
            aver_y(j) = aver_y(j) + fitresult{i}(cal_list(j));
            aver_sum(j) = aver_sum(j)+1;
        end
    end
end

aver_y = aver_y./aver_sum;

class = 1;show = true;
[fitresult_aver, gof_aver] = createFit(cal_list, aver_y ,class, i ,show);

%% 计算每个样本残差
yData_pre = [];
% 计算预测值
for i=2:length(ED_time)
    xData = ED_time(i,:);
    yData = ED_value(i,:);
    ED_count_i = ED_count(i);
    for j = 1:ED_count_i
        % yData_pre_j = fitresult_aver.p1*xData(j)^7 ...
        %             + fitresult_aver.p2*xData(j)^6 ...
        %             + fitresult_aver.p3*xData(j)^5 ...
        %             + fitresult_aver.p4*xData(j)^4 ...
        %             + fitresult_aver.p5*xData(j)^3 ...
        %             + fitresult_aver.p6*xData(j)^2 ...
        %             + fitresult_aver.p7*xData(j)^1 ...
        %             + fitresult_aver.p8;
        yData_pre_j = fitresult_aver(cal_list(j));
        yData_pre(i,j) = yData_pre_j;
    end

end
% 计算误差
ED_err = [];
for i=2:length(ED_time)
    xData = ED_time(i,:);
    yData = ED_value(i,:);
    ED_count_i = ED_count(i);
    for j = 1:ED_count_i
        ED_err(i, j) = yData(j)- yData_pre(i,j);
    end

end
% 误差绝对值
ED_err_abs = abs(ED_err);
sub_cancha_1_100 = sum(ED_err_abs, 2);
figure(20)
plot(sub_cancha_1_100)










function [fitresult, gof] = createFit(X, Y, class, i, show)
[xData, yData] = prepareCurveData( X, Y);

% 设置 fittype 和选项。
ft = fittype( 'poly7' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'LAR';

% % 设置 fittype 和选项。
% ft = fittype( 'gauss1' );
% opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
% opts.DiffMaxChange = 0.5;
% opts.Display = 'Off';
% opts.Lower = [-Inf -Inf 0];
% opts.MaxFunEvals = 60000;
% opts.MaxIter = 40000;
% opts.Robust = 'LAR';
% opts.StartPoint = [15231 26.7653602430556 23.3221951328267];
% opts.TolFun = 1;
% opts.TolX = 1;


% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, opts );

% % 设置 fittype 和选项。
% ft = fittype( 'poly8' )*;
% opts = fitoptions( 'Method', 'LinearLeastSquares' );
% opts.Normalize = 'on';
% opts.Robust = 'LAR';
% 
% % 对数据进行模型拟合。
% [fitresult, gof] = fit( xData, yData, ft, opts );

% 为绘图创建一个图窗。
if(show)
    figure(8+class);hold on;
    style = {'*','+','-','--','.','o'};
    % 绘制数据拟合图。
    subplot( 2, 1, 1 );hold on;
    h = plot( fitresult, xData, yData ,'-');
    legend( h, 'Y vs. X', '无标题拟合 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % 为坐标区加标签
    xlabel( 'X', 'Interpreter', 'none' );
    ylabel( 'Y', 'Interpreter', 'none' );
    grid on;


    % 绘制残差图。
    subplot( 2, 1, 2 );hold on;
    h = plot( fitresult, xData, yData, 'residuals');
    legend( h, '无标题拟合 1 - 残差', 'Zero Line', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % 为坐标区加标签
    xlabel( 'X', 'Interpreter', 'none' );
    ylabel( 'Y', 'Interpreter', 'none' );
    grid on
end
end
