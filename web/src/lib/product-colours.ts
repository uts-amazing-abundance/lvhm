/** Display colour per product, shared by the cell views and the pick Gantt. */
export const PRODUCT_COLOURS: Record<string, string> = {
  P1: '#2a78d6', // Cobalt Blue
  P2: '#eb6834', // Safety Orange
  P3: '#1baf7a', // Jade
  P4: '#d9a21b', // Mustard
  P5: '#8e5bd6', // Violet
  P6: '#d03b3b', // Signal Red
};

export const productColour = (product: string): string => PRODUCT_COLOURS[product] ?? '#898781';
