import FolderPicker from './components/utilities/FolderPicker'

function App() {
  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-8">
      <FolderPicker
        size={1.3}
        color="#5227FF"
        onSelectionChange={(files) => console.log('Selected:', files)}
      />
    </div>
  )
}

export default App