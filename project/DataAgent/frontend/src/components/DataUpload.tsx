import { useState, useRef } from 'react';
import { Button } from './ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { ScrollArea } from './ui/scroll-area';
import { Upload, Trash2 } from 'lucide-react';
import { TelcoData } from '../data/mockData';
import { API_ENDPOINTS } from '../config/api';

interface DataUploadProps {
  onDataUpload: (data: TelcoData[]) => void;
  onDataClear: () => void;
  uploadedData: TelcoData[] | null;
}

export function DataUpload({ onDataUpload, onDataClear, uploadedData }: DataUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (file: File) => {
    if (!file.name.endsWith('.csv')) {
      alert('请上传CSV格式的文件');
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(API_ENDPOINTS.UPLOAD, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === 'success' && result.preview && Array.isArray(result.preview)) {
        onDataUpload(result.preview);
        if (result.message) {
          alert(result.message);
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('文件上传失败，请检查后端服务是否正常运行');
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleClearData = async () => {
    onDataClear();
  };

  if (!uploadedData) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileInputChange}
          className="hidden"
        />
        <div
          className={`w-full max-w-sm p-8 border-2 border-black text-center transition-all duration-150 ${
            isDragOver 
              ? 'bg-secondary border-2' 
              : 'bg-[#D1D1D1] hover:bg-secondary/50'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="w-16 h-16 bg-primary rounded-none flex items-center justify-center mx-auto mb-6 border-2 border-black">
            <Upload className="w-8 h-8 text-white" />
          </div>
          <h4 className="mb-3 text-black font-bold text-lg">上传数据集</h4>
          <p className="text-gray-600 text-sm mb-6 font-medium">
            支持CSV格式，拖拽或点击上传
          </p>
          <Button 
            onClick={handleFileUpload}
            disabled={isUploading}
            className="bg-primary hover:bg-primary/90"
          >
            {isUploading ? '上传中...' : '选择文件'}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col" style={{ backgroundColor: '#D1D1D1' }}>
      <div className="p-4 flex items-center justify-between border-b-2 border-black bg-secondary">
        <h3 className="text-black font-bold text-lg">
          部分数据展示（前10行）
        </h3>
        <Button
          onClick={handleClearData}
          variant="destructive"
          size="sm"
          className="h-8 w-8 p-0"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </Button>
      </div>

      <div className="flex-1 overflow-hidden" style={{ backgroundColor: '#D1D1D1' }}>
        <ScrollArea className="h-full custom-scrollbar">
          <div className="bg-[#D1D1D1] m-3 border-2 border-black overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="border-black hover:bg-secondary/30">
                {uploadedData.length > 0 && Object.keys(uploadedData[0]).map((key) => (
                  <TableHead key={key} className="text-black text-xs bg-secondary/50">
                    {key}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {uploadedData.slice(0, 10).map((row, index) => (
                <TableRow key={index} className="border-black hover:bg-secondary/30 transition-colors">
                  {Object.entries(row).map(([key, value], cellIndex) => (
                    <TableCell key={cellIndex} className="text-xs text-black font-medium">
                      {key === 'Churn' && (value === 'Yes' || value === 'No') ? (
                        <span className={`px-2 py-1 text-xs font-bold border-2 border-black ${
                          value === 'Yes' 
                            ? 'bg-destructive text-white' 
                            : 'bg-accent text-white'
                        }`}>
                          {value}
                        </span>
                      ) : (
                        String(value)
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}