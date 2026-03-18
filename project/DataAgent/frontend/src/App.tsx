import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { ChatInterface } from './components/ChatInterface';
import { DataUpload } from './components/DataUpload';
import { VisualizationPanel } from './components/VisualizationPanel';
import { TelcoData } from './data/mockData';

export default function App() {
  const [uploadedData, setUploadedData] = useState<TelcoData[] | null>(null);
  const [chatClearTrigger, setChatClearTrigger] = useState(0);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);

  const handleDataUpload = (data: TelcoData[]) => {
    setUploadedData(data);
  };

  const handleDataClear = () => {
    setUploadedData(null);
  };

  const handleChatClear = () => {
    setChatClearTrigger(prev => prev + 1);
    setGeneratedImage(null);
  };

  const handleImageGenerated = (imageUrl: string) => {
    setGeneratedImage(imageUrl);
  };

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#f5f0e8]">
      <div className="flex flex-col h-full">
        <Header onChatClear={handleChatClear} />
        
        <div className="flex-1 flex overflow-hidden min-h-0">
          {/* Left Panel - Chat Interface (Area 1) */}
          <div className="w-1/2 flex flex-col min-h-0 border-r-2 border-black bg-white overflow-hidden">
            <ChatInterface clearTrigger={chatClearTrigger} onImageGenerated={handleImageGenerated} />
          </div>
          
          {/* Right Panel - Data Analysis (Area 2 & 3) */}
          <div className="w-1/2 flex flex-col min-h-0">
            {/* Top Half - Visualization (Area 2) */}
            <div className="flex-1 border-b-2 border-black overflow-hidden" style={{ backgroundColor: '#F0F0F0' }}>
              <VisualizationPanel uploadedData={uploadedData} generatedImage={generatedImage} />
            </div>
            
            {/* Bottom Half - Data Upload/Display (Area 3) */}
            <div className="flex-1 overflow-hidden" style={{ backgroundColor: '#D1D1D1' }}>
              <DataUpload
                onDataUpload={handleDataUpload}
                onDataClear={handleDataClear}
                uploadedData={uploadedData}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}