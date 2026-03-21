import { useState, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ReferenceLine } from 'recharts';
import { TelcoData, columnInfo, mockAnalysisResults } from '../data/mockData';
import { BarChart3, TrendingUp, Activity, Image as ImageIcon, RefreshCw } from 'lucide-react';

interface VisualizationPanelProps {
  uploadedData: TelcoData[] | null;
  generatedImage?: string | null;
}

export function VisualizationPanel({ uploadedData, generatedImage }: VisualizationPanelProps) {
  const [variable1, setVariable1] = useState<string>('');
  const [variable2, setVariable2] = useState<string>('');
  const [chartData, setChartData] = useState<any[]>([]);
  const [analysisResult, setAnalysisResult] = useState<{ correlation: number; conclusion: string } | null>(null);
  const [showInteractiveChart, setShowInteractiveChart] = useState<boolean>(false);

  const allColumns = [...columnInfo.categorical, ...columnInfo.numerical];
  
  const isNumerical = (column: string) => columnInfo.numerical.includes(column);
  const isCategorical = (column: string) => columnInfo.categorical.includes(column);

  useEffect(() => {
    if (generatedImage) {
      setShowInteractiveChart(false);
    }
  }, [generatedImage]);

  useEffect(() => {
    if (variable1 && variable2 && uploadedData) {
      generateChartData();
      getAnalysisResult();
    }
  }, [variable1, variable2, uploadedData]);

  const generateChartData = () => {
    if (!uploadedData || !variable1 || !variable2) return;

    if (isCategorical(variable1) && isCategorical(variable2)) {
      const groupedData = uploadedData.reduce((acc, row) => {
        const key = row[variable1 as keyof TelcoData] as string;
        const value = row[variable2 as keyof TelcoData] as string;
        
        if (!acc[key]) acc[key] = {};
        if (!acc[key][value]) acc[key][value] = 0;
        acc[key][value]++;
        
        return acc;
      }, {} as Record<string, Record<string, number>>);

      const chartData = Object.keys(groupedData).map(key => {
        const item: any = { name: key };
        Object.keys(groupedData[key]).forEach(value => {
          item[value] = groupedData[key][value];
        });
        return item;
      });

      setChartData(chartData);
    } else if (isNumerical(variable1) && isNumerical(variable2)) {
      const scatterData = uploadedData.map((row, index) => ({
        x: row[variable1 as keyof TelcoData] as number,
        y: row[variable2 as keyof TelcoData] as number,
        id: index
      }));

      setChartData(scatterData);
    } else {
      const categoricalVar = isCategorical(variable1) ? variable1 : variable2;
      const numericalVar = isNumerical(variable1) ? variable1 : variable2;
      
      const groupedData = uploadedData.reduce((acc, row) => {
        const category = row[categoricalVar as keyof TelcoData] as string;
        const value = row[numericalVar as keyof TelcoData] as number;
        
        if (!acc[category]) acc[category] = [];
        acc[category].push(value);
        
        return acc;
      }, {} as Record<string, number[]>);

      const chartData = Object.keys(groupedData).map(category => {
        const values = groupedData[category];
        return {
          name: category,
          average: values.reduce((sum, val) => sum + val, 0) / values.length,
          min: Math.min(...values),
          max: Math.max(...values),
          count: values.length
        };
      });

      setChartData(chartData);
    }
  };

  const getAnalysisResult = () => {
    const key1 = `${variable1}_${variable2}`;
    const key2 = `${variable2}_${variable1}`;
    
    const result = mockAnalysisResults[key1 as keyof typeof mockAnalysisResults] || 
                   mockAnalysisResults[key2 as keyof typeof mockAnalysisResults];
    
    if (result) {
      setAnalysisResult(result);
    } else {
      const correlation = Math.random() * 2 - 1;
      setAnalysisResult({
        correlation: Number(correlation.toFixed(2)),
        conclusion: `${variable1}与${variable2}之间的相关性为${Math.abs(correlation) > 0.5 ? '强' : Math.abs(correlation) > 0.3 ? '中等' : '弱'}`
      });
    }
  };

  const renderChart = () => {
    if (!chartData.length || !variable1 || !variable2) return null;

    if (isCategorical(variable1) && isCategorical(variable2)) {
      const uniqueValues = Array.from(new Set(
        uploadedData?.map(row => row[variable2 as keyof TelcoData] as string) || []
      ));
      
      const colors = ['#FF6B35', '#FFD166', '#06B6D4', '#EF476F', '#06D6A0'];

      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="none" stroke="#000" strokeWidth={2} />
            <XAxis dataKey="name" stroke="#000" fontSize={11} tick={{ fill: '#000', fontWeight: 700 }} />
            <YAxis stroke="#000" fontSize={11} tick={{ fill: '#000', fontWeight: 700 }} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '2px solid #000',
                borderRadius: '0px',
                color: '#000',
                fontWeight: 600,
                boxShadow: 'none'
              }} 
            />
            {uniqueValues.map((value, index) => (
              <Bar 
                key={value} 
                dataKey={value} 
                stackId="a" 
                fill={colors[index % colors.length]} 
                stroke="#000"
                strokeWidth={2}
                radius={0}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      );
    } else if (isNumerical(variable1) && isNumerical(variable2)) {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="none" stroke="#000" strokeWidth={2} />
            <XAxis dataKey="x" name={variable1} stroke="#000" fontSize={11} tick={{ fill: '#000', fontWeight: 700 }} />
            <YAxis dataKey="y" name={variable2} stroke="#000" fontSize={11} tick={{ fill: '#000', fontWeight: 700 }} />
            <Tooltip 
              cursor={{ strokeDasharray: 'none', stroke: '#000', strokeWidth: 2 }}
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '2px solid #000',
                borderRadius: '0px',
                color: '#000',
                fontWeight: 600,
                boxShadow: 'none'
              }} 
            />
            <Scatter fill="#FF6B35" stroke="#000" strokeWidth={2} />
            <ReferenceLine stroke="#000" strokeDasharray="none" strokeWidth={2} />
          </ScatterChart>
        </ResponsiveContainer>
      );
    } else {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="none" stroke="#000" strokeWidth={2} />
            <XAxis dataKey="name" stroke="#000" fontSize={11} tick={{ fill: '#000', fontWeight: 700 }} />
            <YAxis stroke="#000" fontSize={11} tick={{ fill: '#000', fontWeight: 700 }} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '2px solid #000',
                borderRadius: '0px',
                color: '#000',
                fontWeight: 600,
                boxShadow: 'none'
              }} 
            />
            <Bar dataKey="average" fill="#FF6B35" stroke="#000" strokeWidth={2} radius={0} />
          </BarChart>
        </ResponsiveContainer>
      );
    }
  };

  const getChartTitle = () => {
    if (!variable1 || !variable2) return '请选择两个变量进行分析';
    
    if (isCategorical(variable1) && isCategorical(variable2)) {
      return `${variable1} vs ${variable2} - 堆叠柱状图`;
    } else if (isNumerical(variable1) && isNumerical(variable2)) {
      return `${variable1} vs ${variable2} - 散点图`;
    } else {
      return `${variable1} vs ${variable2} - 分组分析`;
    }
  };

  const getChartIcon = () => {
    if (!variable1 || !variable2) {
      return <BarChart3 className="w-3 h-3 text-white" />;
    }
    
    if (isCategorical(variable1) && isCategorical(variable2)) {
      return <BarChart3 className="w-3 h-3 text-white" />;
    } else if (isNumerical(variable1) && isNumerical(variable2)) {
      return <TrendingUp className="w-3 h-3 text-white" />;
    } else {
      return <Activity className="w-3 h-3 text-white" />;
    }
  };

  if (!uploadedData) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-secondary rounded-none flex items-center justify-center mx-auto mb-4 border-2 border-black">
            <BarChart3 className="w-8 h-8 text-black" />
          </div>
          <p className="text-black mb-1 font-bold">请先上传数据集</p>
          <p className="text-gray-600 text-sm font-medium">上传后可进行变量分析</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col" style={{ backgroundColor: '#F0F0F0' }}>
      <div className="p-4 border-b-2 border-black flex-shrink-0 bg-secondary">
        <div className="flex gap-3 items-center">
          <div className="flex-1 grid grid-cols-2 gap-3">
            <Select value={variable1} onValueChange={setVariable1}>
              <SelectTrigger className="bg-white font-medium rounded-none h-9 text-sm">
                <SelectValue placeholder="选择变量 1" />
              </SelectTrigger>
              <SelectContent className="bg-white border-2 border-black">
                {allColumns.map(column => (
                  <SelectItem key={column} value={column} disabled={column === variable2} className="font-medium">
                    {column}
                    <Badge variant="outline" className="ml-2 text-xs border-black">
                      {isNumerical(column) ? '数值' : '分类'}
                    </Badge>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={variable2} onValueChange={setVariable2}>
              <SelectTrigger className="bg-white font-medium rounded-none h-9 text-sm">
                <SelectValue placeholder="选择变量 2" />
              </SelectTrigger>
              <SelectContent className="bg-white border-2 border-black">
                {allColumns.map(column => (
                  <SelectItem key={column} value={column} disabled={column === variable1} className="font-medium">
                    {column}
                    <Badge variant="outline" className="ml-2 text-xs border-black">
                      {isNumerical(column) ? '数值' : '分类'}
                    </Badge>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {generatedImage && (
            <Button
              onClick={() => setShowInteractiveChart(!showInteractiveChart)}
              className="bg-white text-black hover:bg-secondary rounded-none h-9 px-3 border-2 border-black"
              variant="outline"
            >
              {showInteractiveChart ? (
                <>
                  <ImageIcon className="w-4 h-4 mr-1" />
                  <span className="text-xs font-bold">AI图表</span>
                </>
              ) : (
                <>
                  <BarChart3 className="w-4 h-4 mr-1" />
                  <span className="text-xs font-bold">交互图表</span>
                </>
              )}
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 p-4 overflow-hidden">
        <div className="h-full custom-scrollbar overflow-auto">
          {generatedImage && !showInteractiveChart ? (
            <div className="h-full flex items-center justify-center bg-[#F0F0F0] border-2 border-black p-4">
              <div className="w-full h-full flex flex-col">
                <div className="flex items-center gap-2 mb-3 px-2">
                  <div className="w-6 h-6 bg-primary rounded-none flex items-center justify-center border-2 border-black">
                    <ImageIcon className="w-3 h-3 text-white" />
                  </div>
                  <span className="text-black text-sm font-bold">AI 生成图表</span>
                </div>
                <div className="flex-1 flex items-center justify-center">
                  <img 
                    src={generatedImage} 
                    alt="AI Generated Chart" 
                    className="max-w-full max-h-full object-contain border-2 border-black"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                      e.currentTarget.parentElement?.insertAdjacentHTML('afterbegin', 
                        '<div class="text-black text-center"><p class="mb-2 font-bold">图片加载失败</p><p class="text-sm font-medium">' + generatedImage + '</p></div>'
                      );
                    }}
                  />
                </div>
              </div>
            </div>
          ) : variable1 && variable2 ? (
            <div className="h-full flex gap-4">
              <div className="flex-1 bg-[#F0F0F0] border-2 border-black p-4 overflow-hidden">
                {renderChart()}
              </div>
              
              {analysisResult && (
                <div className="w-64 bg-[#F0F0F0] border-2 border-black p-4 overflow-hidden">
                  <div className="flex flex-col h-full">
                    <div className="custom-scrollbar overflow-y-auto flex-1">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-6 h-6 bg-primary rounded-none flex items-center justify-center border-2 border-black">
                          {getChartIcon()}
                        </div>
                        <span className="text-black text-sm font-bold">分析结果</span>
                      </div>
                      
                      <div className="space-y-4">
                        <div className="bg-secondary border-2 border-black p-3">
                          <div className="text-xs text-black mb-1 font-bold">相似性系数</div>
                          <div className={`text-lg font-bold ${
                            Math.abs(analysisResult.correlation) > 0.5 ? 'text-destructive' : 
                            Math.abs(analysisResult.correlation) > 0.3 ? 'text-primary' : 'text-accent'
                          }`}>
                            {analysisResult.correlation}
                          </div>
                        </div>
                        
                        <div className="bg-[#F0F0F0] border-2 border-black p-3">
                          <div className="text-xs text-black mb-1 font-bold">分析结论</div>
                          <div className="text-sm text-black leading-relaxed font-medium">{analysisResult.conclusion}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 bg-secondary rounded-none flex items-center justify-center mx-auto mb-4 border-2 border-black">
                  <BarChart3 className="w-8 h-8 text-black" />
                </div>
                <p className="text-black font-bold">请选择两个变量开始分析</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}