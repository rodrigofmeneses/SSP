import express from "express";
import { AppDataSource } from "./data-source";

AppDataSource.initialize().then(() => {
  const app = express();

  app.use(express.json());
  app.get("/", (req, res) => {
    res.json({ message: "CheckMate" });
  });

  return app.listen(process.env.PORT, () => {
    console.log(`Servidor escutando em http://localhost:${process.env.PORT}`);
  });
});
