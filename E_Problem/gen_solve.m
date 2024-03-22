clear;clc;
ED_time = readtable("ED time.xlsx");
ED_value = readtable("ED value.xlsx");
ED_Count = readtable("ED Count.xlsx");
ED_time_data = table2array(ED_time(2:end,2:end));
ED_value_data = table2array(ED_value(2:end,2:end));
ED_Count_data = table2array(ED_Count(2:end,2:end));
figure(1);hold on;
syms a b c d e f x
parsnum = 4;
g_fun= c*x.^3+d*x.^2+e*x+f;
f = 0;
[r,c] = size(ED_time_data);  % 行扫描
for i = 1:r
    x_each = 0;y_each = 0;
    for j = 1:ED_Count_data(i)  %计算全部的损失函数
        x_value = ED_time_data(i,j);
        y_value = ED_value_data(i,j);
        x_each(j) = x_value;y_each(j) = y_value;
        f = f + (subs(g_fun,x,x_value) -y_value)^2;
    end
    plot(x_each,y_each);hold on;
end
Opti_replace = {['c'],['d'],['e'],['f']}
for i = 1:parsnum
    Opti_new(i)={['x',num2str(i)]};  %替换其他占空比为x
end
fun = matlabFunction(subs(f,Opti_replace,Opti_new));
fun = @(x)fun(x(1),x(2),x(3),x(4));

x0= rand([1,parsnum])*100; %优化初值
lb=-9999*x0;
ub=repmat(9999,[1,4]);
format short
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point', ...
    'ConstraintTolerance',1e-30,'FunctionTolerance',1e-30,'StepTolerance',1e-30, ...
    'MaxFunctionEvaluations',1e30);
% options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');
disp("遗传算法求解");
VarMin = -9999;   % 决策变量的下界
VarMax = 9999;   % 决策变量的上界
nVar = length(lb);       % 决策变量的数量
VarSize = [1 nVar];  % 决策变量的维度
MaxIt = 10000;        % 最大迭代次数
nPop = 2000;          % 种群大小
pc = 0.8;           % 交叉概率
nc = 2*round(pc*nPop/2);    % 交叉个数
pm = 0.3;           % 变异概率
nm = round(pm*nPop);        % 变异个数
mu = 0.2;           % 变异系数
% VoltageConstraint = false;
% if(VoltageConstraint)
%     [BestSol, BestCosts] =genetic_algorithm(fun_Penalty, nVar, VarSize, VarMin, VarMax, MaxIt, nPop, pc,nc, pm,nm, mu, options);
% else
%     [BestSol, BestCosts] =genetic_algorithm(fun, nVar, VarSize, VarMin, VarMax, MaxIt, nPop, pc,nc, pm,nm, mu, options);
% end

disp("全局搜索算法求解");
    problem = createOptimProblem('fmincon',...
        'objective',fun,...
        'x0',x0,'lb',-50,...
        'ub',50,...
        'nonlcon',[],...
        'options',options);
    gs = GlobalSearch;
    [p,G] = run(gs,problem);

% % 绘制结果图像
% figure();
% semilogy(BestCosts+1, 'LineWidth', 2);
% xlabel('迭代次数');
% ylabel('最优解');
% grid on;
% 
% p=BestSol.Position;G=BestSol.Cost;
% vpa(p,7)  % Optimal
% if(VoltageConstraint==false)
%     p(1)=50;
% end

x = 0:4000;
y = p(1)*x.^3+p(2)*x.^2+p(3)*x.^1+p(4);
plot(x,y)