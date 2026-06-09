import FolderPicker from './components/utilities/FolderPicker'
import { PromptInputBox } from './components/utilities/PromptBox'

function App() {
  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-8">
      <FolderPicker
        size={1.3}
        color="#5227FF"
        onSelectionChange={(files) => console.log('Selected:', files)}
      />



    <div className="flex w-full h-screen items-center justify-center bg-[#0a0a0a]">
      <div className="p-4 w-[500px]">
        <PromptInputBox
          onSend={(msg, files) => {
            alert(`"[Darwin] Submit:", ${msg}, ${files}`);
          }}
          isLoading={false}
          placeholder="Ask Darwin anything..."
        />
      </div>
    </div>


    </div>
  )
}

export default App